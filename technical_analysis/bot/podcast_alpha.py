"""
Podcast Alpha Bot
=================
Monitors finance/investing podcasts for new episodes on YouTube, fetches
transcripts, and uses Claude to extract actionable trade ideas.
Posts to a dedicated Discord channel.

Podcasts tracked:
  - All-In Podcast (Chamath, Jason, Sacks, Friedberg)
  - BG2 Pod (Brad Gerstner, Bill Gurley)
  - Invest Like the Best (Patrick O'Shaughnessy)
  - Dumb Money Live (Chris Camillo)
  - TBPN (John Coogan, Jordi Hays)

Setup:
  1. pip install youtube-transcript-api feedparser  (already done)
  2. In Discord: create a new channel (e.g. #podcast-alpha)
     → Edit Channel → Integrations → Webhooks → New Webhook → copy URL
  3. Add to .env:
       JK_DISCORD_PODCAST_WEBHOOK=https://discord.com/api/webhooks/...
  4. Schedule with launchd:
       sudo cp com.jkbot.podcast-alpha.plist ~/Library/LaunchAgents/
       launchctl load ~/Library/LaunchAgents/com.jkbot.podcast-alpha.plist

Run manually:
  python -m technical_analysis.bot.podcast_alpha
  python -m technical_analysis.bot.podcast_alpha --force VIDEO_ID  # re-analyze a specific video
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env", override=True)

import feedparser
import requests

from technical_analysis.bot.llm_client import llm_chat, llm_chat_json_array


# ---------------------------------------------------------------------------
# Podcast sources  (all channel IDs verified 2026-03-22)
# ---------------------------------------------------------------------------

PODCASTS = [
    {
        "name": "All-In Podcast",
        "youtube_channel_id": "UCESLZhusAkFfsNsApnjF_Cg",
        "hosts": "Chamath Palihapitiya, Jason Calacanis, David Sacks, David Friedberg",
        "emoji": "🎙️",
        "focus": "venture capital, tech, macro, policy",
    },
    {
        "name": "Invest Like the Best",
        "youtube_channel_id": "UCpQBb0fToph3jrDulwz1iUQ",
        "hosts": "Patrick O'Shaughnessy",
        "emoji": "💡",
        "focus": "deep-dive interviews with investors and founders",
    },
    {
        "name": "Dumb Money Live",
        "youtube_channel_id": "UCS01CiRDAiyhR_mTHXDW23A",
        "hosts": "Chris Camillo, Dave Hanson, Jordan McLain",
        "emoji": "🎲",
        "focus": "social arbitrage, retail trend detection, consumer stocks",
    },
    {
        "name": "TBPN",
        "youtube_channel_id": "UC-DRzaGnL_vtBUpCFH5M0tg",
        "hosts": "John Coogan, Jordi Hays",
        "emoji": "📡",
        "focus": "daily tech and startup news, AI, founder news",
    },
]


# ---------------------------------------------------------------------------
# State  (tracks which videos have been processed)
# ---------------------------------------------------------------------------

STATE_FILE = Path(__file__).parent / "state" / "podcast_alpha_state.json"


def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"seen_videos": {}}


def save_state(state: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ---------------------------------------------------------------------------
# YouTube RSS feed polling
# ---------------------------------------------------------------------------

def fetch_latest_videos(podcast: dict, max_videos: int = 3) -> list[dict]:
    """Fetch the latest N videos from a podcast YouTube channel via RSS."""
    rss_url = f"https://www.youtube.com/feeds/videos.xml?channel_id={podcast['youtube_channel_id']}"
    try:
        feed = feedparser.parse(rss_url)
        videos = []
        for entry in feed.entries[:max_videos]:
            # YouTube RSS entries have yt_videoid or it's embedded in the id
            vid_id = getattr(entry, "yt_videoid", None) or entry.id.split("v=")[-1]
            videos.append({
                "id": vid_id,
                "title": entry.title,
                "published": entry.get("published", ""),
                "url": f"https://www.youtube.com/watch?v={vid_id}",
                "podcast": podcast["name"],
                "podcast_emoji": podcast["emoji"],
                "hosts": podcast["hosts"],
                "focus": podcast["focus"],
            })
        return videos
    except Exception as e:
        print(f"  [podcast_alpha] RSS error for {podcast['name']}: {e}")
        return []


# ---------------------------------------------------------------------------
# Transcript fetching
# ---------------------------------------------------------------------------

def _truncate_transcript(text: str, max_chars: int) -> str:
    """Truncate a transcript to max_chars, keeping beginning and end."""
    if len(text) <= max_chars:
        return text
    return (
        text[:60_000]
        + "\n\n[...middle omitted...]\n\n"
        + text[-20_000:]
    )


def fetch_transcript(
    video_id: str,
    max_chars: int = 80_000,
    min_chars: int = 5_000,
) -> Optional[str]:
    """
    Fetch YouTube transcript using three methods in order:

    1. Page-scrape: fetch the video page HTML, extract captionTracks JSON,
       request the caption URL with &fmt=json3 (JSON format, avoids XML parsing
       issues). Falls back to XML parsing if JSON unavailable.
    2. youtube-transcript-api library.
    3. yt-dlp subtitle download (most resilient against IP blocks).

    Returns plain text, or None if unavailable / too short (clips/shorts).
    """
    import re
    import html as _html

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    def _check_length(text: str, method: str) -> Optional[str]:
        """Return truncated text if long enough, None if too short."""
        if len(text) >= min_chars:
            return _truncate_transcript(text, max_chars)
        print(f"  [podcast_alpha] Clip/short ({video_id}): {len(text):,} chars via {method}, skipping")
        return None

    def _parse_json3_transcript(json_text: str) -> Optional[str]:
        """Parse YouTube's JSON3 timedtext format into plain text."""
        import json as _json
        try:
            data = _json.loads(json_text)
            events = data.get("events", [])
            parts = []
            for event in events:
                segs = event.get("segs", [])
                for seg in segs:
                    raw = seg.get("utf8", "")
                    if raw and raw != "\n":
                        parts.append(_html.unescape(raw).replace("\n", " ").strip())
            return " ".join(p for p in parts if p) if parts else None
        except (ValueError, KeyError):
            return None

    def _parse_xml_transcript(xml_text: str) -> Optional[str]:
        """Parse YouTube's timedtext XML into plain text. Returns None on parse failure."""
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(xml_text)
            parts = []
            for text_el in root.iter("text"):
                raw = text_el.text or ""
                parts.append(_html.unescape(raw).replace("\n", " "))
            return " ".join(parts) if parts else None
        except ET.ParseError:
            return None

    def _extract_captions_from_page(page_html: str) -> Optional[str]:
        """Extract transcript text from captionTracks in YouTube page HTML."""
        import json as _json
        match = re.search(r'"captionTracks":(\[.*?\])', page_html)
        if not match:
            return None
        try:
            tracks = _json.loads(match.group(1))
        except (ValueError, _json.JSONDecodeError):
            return None
        # Prefer English; fall back to first available
        en_tracks = [t for t in tracks if t.get("languageCode", "").startswith("en")]
        track = en_tracks[0] if en_tracks else (tracks[0] if tracks else None)
        if not track:
            return None
        caption_url = track.get("baseUrl", "")
        if not caption_url:
            return None

        # Try JSON3 format first (avoids XML parsing issues)
        json3_url = caption_url + ("&" if "?" in caption_url else "?") + "fmt=json3"
        try:
            cr = requests.get(json3_url, headers=headers, timeout=15)
            if cr.status_code == 200:
                text = _parse_json3_transcript(cr.text)
                if text:
                    return text
        except Exception:
            pass

        # Fall back to default format (XML)
        try:
            cr = requests.get(caption_url, headers=headers, timeout=15)
            if cr.status_code == 200:
                text = _parse_xml_transcript(cr.text)
                if text:
                    return text
        except Exception:
            pass

        return None

    # --- Method 1: scrape caption URL from page HTML ---
    try:
        page_url = f"https://www.youtube.com/watch?v={video_id}"
        r = requests.get(page_url, headers=headers, timeout=15)
        r.raise_for_status()
        text = _extract_captions_from_page(r.text)
        if text is not None:
            result = _check_length(text, "page-scrape")
            if result is not None:
                return result
            # Too short = clip/short, return None immediately
            return None
    except Exception as e:
        print(f"  [podcast_alpha] Page-scrape failed ({video_id}): {e} — trying library fallback")

    # --- Method 2: youtube-transcript-api library fallback ---
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        api = YouTubeTranscriptApi()
        segments = api.fetch(video_id)
        text = " ".join(seg.text for seg in segments)
        result = _check_length(text, "yt-transcript-api")
        if result is not None:
            return result
        return None
    except Exception as e:
        print(f"  [podcast_alpha] yt-transcript-api failed ({video_id}): {e} — trying yt-dlp fallback")

    # --- Method 3: yt-dlp subtitle extraction (most resilient) ---
    try:
        import subprocess
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # Try auto-generated English subtitles first, then manual subs
            cmd = [
                "/opt/anaconda3/bin/yt-dlp",
                "--skip-download",
                "--write-auto-sub",
                "--write-sub",
                "--sub-lang", "en",
                "--sub-format", "json3",
                "--output", f"{tmpdir}/%(id)s.%(ext)s",
                f"https://www.youtube.com/watch?v={video_id}",
            ]
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60,
            )
            # Look for the downloaded subtitle file
            import glob
            sub_files = glob.glob(f"{tmpdir}/*.json3") + glob.glob(f"{tmpdir}/*.en.json3")
            if not sub_files:
                # Try vtt format as fallback
                cmd_vtt = [
                    "/opt/anaconda3/bin/yt-dlp",
                    "--skip-download",
                    "--write-auto-sub",
                    "--write-sub",
                    "--sub-lang", "en",
                    "--sub-format", "vtt",
                    "--output", f"{tmpdir}/%(id)s.%(ext)s",
                    f"https://www.youtube.com/watch?v={video_id}",
                ]
                subprocess.run(cmd_vtt, capture_output=True, text=True, timeout=60)
                sub_files = glob.glob(f"{tmpdir}/*.vtt")

            if not sub_files:
                print(f"  [podcast_alpha] yt-dlp found no subtitles ({video_id})")
                return None

            sub_path = sub_files[0]
            with open(sub_path) as f:
                content = f.read()

            # Parse based on format
            if sub_path.endswith(".json3"):
                text = _parse_json3_transcript(content)
            else:
                # VTT format: strip timing lines, keep text
                lines = []
                for line in content.split("\n"):
                    line = line.strip()
                    # Skip VTT headers, timing lines, and blank lines
                    if not line or line.startswith("WEBVTT") or line.startswith("Kind:") or line.startswith("Language:"):
                        continue
                    if re.match(r"^\d{2}:\d{2}:\d{2}\.\d{3}\s*-->", line):
                        continue
                    if re.match(r"^\d+$", line):  # sequence numbers
                        continue
                    # Strip VTT tags like <c> </c> <00:00:01.234>
                    cleaned = re.sub(r"<[^>]+>", "", line)
                    cleaned = _html.unescape(cleaned).strip()
                    if cleaned:
                        lines.append(cleaned)
                # Deduplicate consecutive identical lines (common in auto-subs)
                deduped = []
                for line in lines:
                    if not deduped or line != deduped[-1]:
                        deduped.append(line)
                text = " ".join(deduped) if deduped else None

            if text:
                result = _check_length(text, "yt-dlp")
                if result is not None:
                    return result
            return None
    except Exception as e:
        print(f"  [podcast_alpha] yt-dlp failed ({video_id}): {e}")
        return None


# ---------------------------------------------------------------------------
# LLM analysis
# ---------------------------------------------------------------------------

ANALYSIS_SYSTEM_PROMPT = """You are a sharp buy-side equity analyst. Your job is to listen to a podcast \
transcript and extract only the specific, actionable trade ideas discussed — the kind of insight that \
would move a position if acted on within 48 hours of episode release.

For each idea output:
  ticker:     specific stock/ETF symbol, or "SECTOR: XYZ" if no ticker named
  direction:  LONG or SHORT
  conviction: HIGH, MEDIUM, or LOW
  horizon:    IMMEDIATE (1-5 days), SHORT (weeks-months), or STRUCTURAL (1+ years)
  thesis:     1 sentence — the actual alpha insight, not a generic view
  source:     who said it, any named catalyst, timeframe, or data point

Conviction guide:
  HIGH   = specific, non-consensus, time-sensitive, named catalyst — act within days
  MEDIUM = directional view with clear reasoning, non-trivial
  LOW    = mentioned in passing, speculative, or very early stage

Rules:
  - Only ideas where a specific company, ticker, or sector was explicitly named
  - No generic market commentary ("bullish on AI", "rates will fall")
  - No ideas that are already consensus/fully priced
  - 8 ideas max — ruthlessly filter for quality over quantity

Output format: a JSON array. Example:
[
  {
    "ticker": "PANW",
    "direction": "LONG",
    "conviction": "HIGH",
    "horizon": "SHORT",
    "thesis": "RSA conference next week is a catalyst; Chamath cited partner checks showing Cortex pull-through in enterprise renewals ahead of expectations",
    "source": "Chamath — referenced 3 portfolio company CISOs switching to Cortex"
  }
]

If there are zero actionable ideas in this episode, respond with exactly: NO_ACTIONABLE_IDEAS"""


def analyze_transcript(
    transcript: str,
    podcast_name: str,
    episode_title: str,
    hosts: str,
    focus: str,
) -> Optional[list[dict]]:
    """Use local Ollama (Qwen 3 4B) to extract trade ideas from a transcript. Returns list or None on error."""
    user_msg = (
        f"Podcast: {podcast_name}\n"
        f"Hosts: {hosts}\n"
        f"Focus area: {focus}\n"
        f"Episode title: {episode_title}\n\n"
        f"TRANSCRIPT:\n{transcript}\n\n"
        "Extract all actionable trade ideas. Be selective — only genuine alpha.\n"
        "If no actionable ideas, return an empty array []."
    )

    try:
        ideas = llm_chat_json_array(
            system=ANALYSIS_SYSTEM_PROMPT,
            user=user_msg,
            max_tokens=2000,
            temperature=0.3,
        )
        return ideas if isinstance(ideas, list) else []
    except Exception as e:
        print(f"  [podcast_alpha] LLM analysis error: {e}")
        return None


# ---------------------------------------------------------------------------
# Discord formatting
# ---------------------------------------------------------------------------

CONVICTION_EMOJI = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "⚪"}
DIRECTION_EMOJI  = {"LONG": "📈", "SHORT": "📉"}


def format_podcast_embed(video: dict, ideas: list[dict]) -> dict:
    """Format extracted trade ideas as a Discord embed payload."""
    emoji = video["podcast_emoji"]
    ep_title = video["title"]
    ep_url   = video["url"]
    hosts    = video["hosts"]
    pod_name = video["podcast"]

    if not ideas:
        return {"embeds": [{
            "title": f"{emoji} {pod_name} — No Actionable Ideas",
            "description": (
                f"**[{ep_title}]({ep_url})**\n\n"
                "Episode analyzed — no specific trade ideas identified."
            ),
            "color": 0x95a5a6,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "footer": {"text": f"JK Podcast Alpha  |  {hosts}"},
        }]}

    fields = []
    for idea in ideas[:8]:  # Discord embed limit
        conv      = idea.get("conviction", "LOW")
        direction = idea.get("direction", "?")
        horizon   = idea.get("horizon", "?")
        ticker    = idea.get("ticker", "?")
        thesis    = idea.get("thesis", "")
        source    = idea.get("source", "")

        field_value = thesis
        if source:
            field_value += f"\n*— {source}*"
        field_value += f"\n`{direction}  ·  {horizon}  ·  {conv}`"

        fields.append({
            "name": (
                f"{CONVICTION_EMOJI.get(conv, '⚪')} "
                f"{DIRECTION_EMOJI.get(direction, '')} "
                f"**{ticker}**"
            ),
            "value": field_value,
            "inline": False,
        })

    high_count = sum(1 for i in ideas if i.get("conviction") == "HIGH")
    color = 0xe74c3c if high_count > 0 else 0xf39c12

    return {"embeds": [{
        "title": (
            f"{emoji} {pod_name} — "
            f"{len(ideas)} Trade Idea{'s' if len(ideas) != 1 else ''}"
            f"  ({high_count} HIGH)"
        ),
        "description": f"**[{ep_title}]({ep_url})**",
        "color": color,
        "fields": fields,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "footer": {"text": f"JK Podcast Alpha  |  {hosts}"},
    }]}


def post_to_discord(payload: dict, url: str):
    resp = requests.post(url, json=payload, timeout=15)
    resp.raise_for_status()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_podcast_alpha(verbose: bool = True, force_video_id: Optional[str] = None) -> tuple[int, int]:
    """
    Poll all configured podcasts, analyze new episodes, post to Discord.

    Args:
        verbose: print progress to stdout
        force_video_id: if set, remove this video from seen-cache and re-analyze it

    Returns:
        (new_episodes_found, trade_ideas_posted)
    """
    webhook_url = os.environ.get("JK_DISCORD_PODCAST_WEBHOOK")
    if not webhook_url:
        print("  [podcast_alpha] WARNING: JK_DISCORD_PODCAST_WEBHOOK not set — ideas will print but not post to Discord")

    state = load_state()
    seen  = state["seen_videos"]

    # --force: clear a specific video so it gets re-processed
    if force_video_id and force_video_id in seen:
        del seen[force_video_id]
        save_state(state)
        print(f"  [podcast_alpha] Cleared {force_video_id} from seen-cache — will re-analyze")

    new_episodes_found = 0
    ideas_posted_total = 0

    for podcast in PODCASTS:
        if verbose:
            print(f"\n  Checking {podcast['name']}...")

        videos = fetch_latest_videos(podcast, max_videos=3)

        for video in videos:
            vid_id = video["id"]

            if vid_id in seen and not (force_video_id and vid_id == force_video_id):
                if verbose:
                    print(f"    Already seen: {video['title'][:65]}")
                continue

            print(f"    NEW EPISODE: {video['title'][:70]}")
            new_episodes_found += 1

            # Mark as seen immediately so a crash mid-analysis doesn't cause duplicate posts
            seen[vid_id] = {
                "title":        video["title"],
                "podcast":      video["podcast"],
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "ideas_found":  None,  # will update after analysis
            }
            save_state(state)

            # Brief pause between transcript requests — avoids YouTube rate-limiting
            # when multiple new episodes appear in the same run
            import time as _time
            _time.sleep(2)

            # Step 1: fetch transcript
            transcript = fetch_transcript(vid_id)
            if not transcript:
                if verbose:
                    print(f"    Transcript not available yet — will retry next run")
                # Remove from seen so we retry next poll
                del seen[vid_id]
                save_state(state)
                continue

            if verbose:
                print(f"    Transcript: {len(transcript):,} chars — analyzing with Claude...")

            # Step 2: LLM analysis
            ideas = analyze_transcript(
                transcript,
                podcast["name"],
                video["title"],
                podcast["hosts"],
                podcast["focus"],
            )
            if ideas is None:
                print(f"    Analysis failed — will retry next run")
                del seen[vid_id]
                save_state(state)
                continue

            # Update seen record with idea count
            seen[vid_id]["ideas_found"] = len(ideas)
            save_state(state)

            if verbose:
                print(f"    Found {len(ideas)} trade idea(s):")
                for idea in ideas:
                    print(
                        f"      [{idea.get('conviction','?')}] "
                        f"{idea.get('direction','?')} {idea.get('ticker','?')} — "
                        f"{idea.get('thesis','')[:80]}"
                    )

            # Step 3: post to Discord (skip if no ideas to keep channel clean)
            if not ideas:
                if verbose:
                    print(f"    No actionable ideas — skipping Discord post")
                continue

            if webhook_url:
                payload = format_podcast_embed(video, ideas)
                try:
                    post_to_discord(payload, webhook_url)
                    ideas_posted_total += len(ideas)
                    if verbose:
                        print(f"    ✓ Posted to Discord")
                except Exception as e:
                    print(f"    Discord post failed: {e}")
            else:
                # No webhook — just print
                for idea in ideas:
                    print(f"      → {idea.get('direction')} {idea.get('ticker')}: {idea.get('thesis')}")

    if verbose:
        print(f"\n  [podcast_alpha] Done — {new_episodes_found} new episode(s), {ideas_posted_total} trade idea(s) posted")

    return new_episodes_found, ideas_posted_total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="JK Podcast Alpha — finance podcast trade idea scanner"
    )
    parser.add_argument(
        "--force", "-f",
        metavar="VIDEO_ID",
        default=None,
        help="Force re-analyze a specific YouTube video ID (clears it from seen-cache)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output",
    )
    args = parser.parse_args()

    run_podcast_alpha(verbose=not args.quiet, force_video_id=args.force)
