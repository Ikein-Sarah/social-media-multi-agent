import streamlit as st
import asyncio
import yt_dlp
import datetime
from agents import Agent, Runner, WebSearchTool, function_tool
from openai import OpenAI
import os
from dotenv import load_dotenv
from typing import List
from dataclasses import dataclass
import requests
from google.oauth2 import service_account
from googleapiclient.discovery import build
import tempfile
import re  # Added for URL extraction

# Load environment variables
load_dotenv(override=True)

# Set page config
st.set_page_config(
    page_title="Social Media Content Generator",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize clients
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Initialize session state
if 'current_content' not in st.session_state:
    st.session_state.current_content = None
if 'notion_link' not in st.session_state:
    st.session_state.notion_link = None
if 'calendar_link' not in st.session_state:
    st.session_state.calendar_link = None
if 'transcript' not in st.session_state:
    st.session_state.transcript = None
if 'evaluation_score' not in st.session_state:
    st.session_state.evaluation_score = None
if 'evaluation_feedback' not in st.session_state:
    st.session_state.evaluation_feedback = None
if 'regenerate_count' not in st.session_state:
    st.session_state.regenerate_count = 0


# Helper function to extract URL from markdown
def extract_url_from_markdown(text):
    """Extract URL from markdown format [text](url)"""
    pattern = r'\[.*?\]\((https?://[^\s]+)\)'
    match = re.search(pattern, text)
    if match:
        return match.group(1)  # Return the URL part
    return text  # Return original text if no markdown link found


# --------------------------------------------------------------
# Your existing functions (with calendar fix)
# --------------------------------------------------------------

@function_tool
def generate_linkedin_post(transcript: str) -> str:
    """Generate LinkedIn post from transcript"""
    prompt = f"""
    Rewrite this TikTok content into a professional LinkedIn post:
    - Engaging but professional
    - Short paragraphs
    - Clear call to action
    - Suitable hashtag

    Content:
    {transcript}
    """

    response = client.responses.create(
        model="gpt-4o",
        instructions="You are an expert LinkedIn post writer",
        input=prompt,
        max_output_tokens=1500
    )
    return response.output_text


content_writer_agent = Agent(
    name="LinkedIn Content Writer Agent",
    instructions=(f"""You are a professional LinkedIn content writer. 
                          Your task is to take the transcript of a TikTok video, 
                          understand the key message, and rewrite it into a polished LinkedIn post. 
                          The post should be engaging, clear, and written in a professional tone 
                          that fits LinkedIn's audience. Avoid slang, keep sentences concise, 
                          and highlight insights or value the audience can take away. 
                          You may search the web for up-to-date information on the topic and fill in some useful details if needed"""),
    tools=[WebSearchTool(), generate_linkedin_post],
    output_type=str,
)


def download_tiktok_audio(url: str) -> str:
    """Download audio from TikTok URL using yt-dlp"""
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(tempfile.gettempdir(), 'tiktok_audio.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            audio_filename = ydl.prepare_filename(info_dict)
            base, ext = os.path.splitext(audio_filename)
            return f"{base}.mp3"
    except Exception as e:
        st.error(f"Download error: {e}")
        raise


def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio using Whisper"""
    try:
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        return transcript.text
    except Exception as e:
        st.error(f"Transcription error: {e}")
        raise


def transcribe_tiktok(url: str) -> str:
    """Main function to handle TikTok URL"""
    try:
        # Download audio
        audio_path = download_tiktok_audio(url)

        # Transcribe
        transcript = transcribe_audio(audio_path)

        # Clean up
        if os.path.exists(audio_path):
            os.remove(audio_path)

        return transcript
    except Exception as e:
        st.error(f"Error processing TikTok: {e}")
        raise


@dataclass
class EvaluationFeedback:
    feedback: str
    score: int


evaluator = Agent[None](
    name="LinkedInEvaluator",
    instructions=(
        """You evaluate a LinkedIn post generated from a TikTok transcript and decide if it's good enough. Give it a score from 1-10.
        By "good enough", I mean the post must sound like it was written by me (the user), and be professional, insightful, and highly readable for LinkedIn.
        If it's not good enough, you provide feedback on what needs to be improved. For example, you should:
        1. Evaluate the most important thing: authenticity. Does it sound like my authentic voice and writing style?
        2. Assess the transformation from casual TikTok tone to a professional, yet personal, LinkedIn voice
        3. Check value-add and depth: does it provide more insight than the original transcript?
        4. Verify factual accuracy and relevance to the original video's key messages
        5. Analyze LinkedIn-specific optimization (format, paragraph length, professional hashtag usage)
        6. Offer specific, actionable recommendations for improvement."""
    ),
    output_type=EvaluationFeedback,
)


@dataclass
class StoredPost:
    platform: str
    filename: str
    filelink: str


@dataclass
class StoredPosts:
    posts: List[StoredPost]


@function_tool
def save_to_notion(content: str, title: str) -> str:
    """
    Save content to Notion automatically.
    """
    NOTION_API_KEY = os.getenv("NOTION_API_KEY")
    NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")

    if not NOTION_API_KEY or not NOTION_DATABASE_ID:
        return "Notion credentials not configured"

    url = "https://api.notion.com/v1/pages"

    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28"
    }

    data = {
        "parent": {"database_id": NOTION_DATABASE_ID},
        "properties": {
            "Title": {
                "title": [
                    {
                        "text": {
                            "content": title[:200]
                        }
                    }
                ]
            },
            "Platform": {
                "select": {
                    "name": "LinkedIn"
                }
            },
            "Status": {
                "select": {
                    "name": "Draft"
                }
            }
        },
        "children": [
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {
                                "content": content
                            }
                        }
                    ]
                }
            }
        ]
    }

    try:
        response = requests.post(url, json=data, headers=headers)

        if response.status_code == 200:
            page_id = response.json()["id"]
            return f"https://www.notion.so/{page_id.replace('-', '')}"
        else:
            return f"Notion Error: {response.text}"

    except Exception as e:
        return f"Notion Connection Failed: {e}"


content_storage_agent = Agent(
    name="LinkedIn Content Storage Agent",
    instructions=f"""Save LinkedIn posts to Notion with clear, descriptive titles.
    Use this naming format: 'LinkedIn Post - [Topic Summary]'
    Make sure the post is saved properly and return the Notion link.""",
    model="gpt-4o-mini",
    tools=[save_to_notion],
    output_type=StoredPosts,
)


# Authenticate Google Calendar API
def authenticate_google_calendar():
    """Authenticate and return Google Calendar service"""
    try:
        SERVICE_ACCOUNT_FILE = "social-media-agent.json"
        SCOPES = ["https://www.googleapis.com/auth/calendar"]

        if not os.path.exists(SERVICE_ACCOUNT_FILE):
            return None

        creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES
        )
        calendar_service = build("calendar", "v3", credentials=creds)
        return calendar_service
    except Exception as e:
        st.error(f"Google Calendar authentication error: {e}")
        return None


calendar_service = authenticate_google_calendar()


@function_tool
def get_time_now_lagos():
    """
    Get the current time in Lagos, Nigeria timezone (Africa/Lagos, UTC+1)
    """
    lagos_timezone = datetime.timezone(datetime.timedelta(hours=1))
    lagos_now = datetime.datetime.now(lagos_timezone)
    return lagos_now


@function_tool
def add_to_calendar(event_title: str, event_description: str, suggested_time: str):
    """
    Adds an event to the calendar with the specified title, description, and post time.
    """
    if not calendar_service:
        return "Google Calendar not configured"

    try:
        suggested_time = datetime.datetime.fromisoformat(suggested_time)

        # Create a shorter description for the calendar event
        short_description = f"Content to post: {event_description[:100]}..." if len(
            event_description) > 100 else event_description

        event = {
            'summary': event_title,
            'description': f"{short_description}\n\nFull content available in Notion.",
            'start': {
                'dateTime': suggested_time.isoformat(),
                'timeZone': 'Africa/Lagos',
            },
            'end': {
                'dateTime': (suggested_time + datetime.timedelta(hours=1)).isoformat(),
                'timeZone': 'Africa/Lagos',
            },
            'reminders': {
                'useDefault': False,
                'overrides': [
                    {'method': 'email', 'minutes': 60},
                    {'method': 'popup', 'minutes': 15},
                ],
            },
        }

        event_result = calendar_service.events().insert(
            calendarId='elosarah85@gmail.com',
            body=event
        ).execute()

        return event_result.get("htmlLink")
    except Exception as e:
        return f"Calendar Error: {e}"


# FIXED: Updated scheduling agent instructions
content_scheduling_agent = Agent(
    name="Social Media Content Scheduling Agent",
    instructions=f"""Schedule ONLY ONE posting event for this LinkedIn content. 
    Choose the next available weekday (Monday-Friday) in the morning (7-8 AM Lagos time).
    DO NOT create multiple events - create only one event for the optimal posting time.
    Include a brief summary of the content in the event description.

    **CRITICAL: Return ONLY the raw calendar URL as a plain text string.**
    **DO NOT format it as markdown, DO NOT add any additional text.**
    **Example: https://www.google.com/calendar/event?eid=abc123**

    Return only the single event link.""",
    model="gpt-4o-mini",
    tools=[get_time_now_lagos, add_to_calendar],
)


# Helper function to extract Notion link
def extract_notion_link(storage_result):
    """Extract the Notion link from StoredPosts object"""
    if hasattr(storage_result, 'posts') and storage_result.posts:
        return storage_result.posts[0].filelink
    elif hasattr(storage_result, 'final_output'):
        return extract_notion_link(storage_result.final_output)
    else:
        return "Error: Could not extract Notion link"


# Async function to process the video
async def process_video(url, regenerate=False):
    """Process TikTok video and generate LinkedIn content"""
    try:
        if not regenerate:
            # Step 1: Transcribe (only on first run)
            st.session_state.transcript = transcribe_tiktok(url)

        # Step 2: Generate post
        msg = f"Generate a LinkedIn post based on this TikTok transcript: {st.session_state.transcript}"
        input_items = [{"content": msg, "role": "user"}]

        # If regenerating, use previous feedback
        if regenerate and st.session_state.evaluation_feedback:
            input_items.append(
                {"content": f"Previous feedback: {st.session_state.evaluation_feedback}", "role": "user"})

        latest_content = None
        for attempt in range(2):  # Max 2 attempts
            # Generate LinkedIn post
            generated_content_result = await Runner.run(
                content_writer_agent,
                input_items,
            )
            latest_content = generated_content_result.final_output

            # Evaluate the generated post
            evaluator_input = [{"content": f"Generated LinkedIn post: {latest_content}", "role": "user"}]
            evaluator_result = await Runner.run(evaluator, evaluator_input)
            result = evaluator_result.final_output

            st.session_state.evaluation_score = result.score
            st.session_state.evaluation_feedback = result.feedback

            # If score is good, break out of the loop
            if result.score >= 8:
                break

            # Add feedback for the next iteration
            input_items.append({"content": f"Feedback: {result.feedback}", "role": "user"})

        # Step 3: Save to Notion
        storage_result = await Runner.run(
            content_storage_agent,
            input=f"Save this LinkedIn post: {latest_content}",
        )
        notion_link = extract_notion_link(storage_result)

        # Step 4: Schedule the post
        schedule_result = await Runner.run(
            content_scheduling_agent,
            input=f"Schedule this LinkedIn post for posting: {latest_content}",
        )
        calendar_link = schedule_result.final_output

        # Extract URL from markdown if needed (backup)
        calendar_link = extract_url_from_markdown(calendar_link)

        return latest_content, notion_link, calendar_link

    except Exception as e:
        st.error(f"Error processing video: {e}")
        return None, None, None


# --------------------------------------------------------------
# Streamlit UI - Clean and Simple
# --------------------------------------------------------------

# Sidebar for platform selection
with st.sidebar:
    st.header("⚙️ Settings")

    platform = st.selectbox(
        "Social Platform",
        ["LinkedIn", "X", "Instagram", "Medium"],
        index=0,
        help="Select the platform for your content"
    )

    posting_time = st.time_input(
        "Preferred Posting Time",
        datetime.time(7, 30),
        help="Best time to schedule your posts (Lagos time)"
    )

    st.divider()
    st.info("""
    **Current Capabilities:**
    - TikTok to LinkedIn conversion
    - Quality evaluation
    - Auto-saving to Notion
    - Calendar scheduling
    """)

# Main content area
st.title("📱 Social Media Content Generator")
st.markdown("Transform TikTok videos into professional social media posts")

# Input section
tiktok_url = st.text_input(
    "TikTok URL",
    placeholder="https://www.tiktok.com/@username/video/1234567890",
    help="Paste the TikTok video URL here"
)

process_btn = st.button(
    "🚀 Generate Content",
    type="primary",
    disabled=not tiktok_url or not OPENAI_API_KEY,
    use_container_width=True
)

# Processing and results
if process_btn and tiktok_url and OPENAI_API_KEY:
    with st.spinner("Processing your TikTok video..."):
        # Run the async processing
        latest_content, notion_link, calendar_link = asyncio.run(process_video(tiktok_url))

        if latest_content:
            st.session_state.current_content = latest_content
            st.session_state.notion_link = notion_link
            st.session_state.calendar_link = calendar_link
            st.session_state.regenerate_count = 0
            st.success("✅ Content generated successfully!")

# Display results
if st.session_state.current_content:
    st.divider()

    # Create tabs for different content views
    tab1, tab2, tab3 = st.tabs(["Generated Content", "Original Transcript", "Evaluation"])

    with tab1:
        st.subheader("📝 Generated LinkedIn Post")
        st.text_area(
            "Content",
            st.session_state.current_content,
            height=200,
            label_visibility="collapsed",
            key="content_display"
        )

        # Regenerate button
        if st.button("🔄 Regenerate Post", type="secondary"):
            st.session_state.regenerate_count += 1
            with st.spinner("Regenerating content..."):
                latest_content, notion_link, calendar_link = asyncio.run(
                    process_video(tiktok_url, regenerate=True)
                )

                if latest_content:
                    st.session_state.current_content = latest_content
                    st.session_state.notion_link = notion_link
                    st.session_state.calendar_link = calendar_link
                    st.rerun()

    with tab2:
        if st.session_state.transcript:
            st.subheader("🎤 Original Transcript")
            st.text_area(
                "Transcript",
                st.session_state.transcript,
                height=300,
                label_visibility="collapsed"
            )
        else:
            st.info("No transcript available")

    with tab3:
        if st.session_state.evaluation_score is not None:
            st.subheader("📊 Quality Evaluation")
            st.metric("Score", f"{st.session_state.evaluation_score}/10")

            if st.session_state.evaluation_feedback:
                with st.expander("View Feedback"):
                    st.write(st.session_state.evaluation_feedback)
        else:
            st.info("No evaluation available")

    # Notion and Calendar links
    st.divider()
    st.subheader("Output")

    col1, col2 = st.columns(2)

    with col1:
        if st.session_state.notion_link and "Error" not in str(st.session_state.notion_link):
            st.link_button("📋 View in Notion", st.session_state.notion_link, use_container_width=True)
        else:
            st.info("Notion link not available")

    with col2:
        if st.session_state.calendar_link and "Error" not in str(st.session_state.calendar_link):
            # Use markdown to open in new tab
            st.markdown(
                f'<a href="{st.session_state.calendar_link}" target="_blank" style="text-decoration: none;">'
                f'<button style="width: 100%; background-color: #FF4B4B; color: white; border: none; padding: 10px; border-radius: 5px; cursor: pointer;">'
                f'📅 View in Calendar'
                f'</button></a>',
                unsafe_allow_html=True
            )
        else:
            st.info("Calendar link not available")

# Instructions
with st.expander("ℹ️ How to use this app"):
    st.markdown("""
    1. **Paste a TikTok URL** in the input field
    2. **Click 'Generate Content'** to create a LinkedIn post
    3. **View your content** in the different tabs:
       - **Generated Content**: Your LinkedIn post
       - **Original Transcript**: The text from the TikTok video
       - **Evaluation**: Quality score and feedback
    4. **Use the regenerate button** if you want to improve the content
    5. **View your content** in Notion and Google Calendar

    **Note:** Make sure your `.env` file contains:
    - `OPENAI_API_KEY` for AI capabilities
    - `NOTION_API_KEY` and `NOTION_DATABASE_ID` for saving to Notion
    - `social-media-agent.json` for Google Calendar integration
    """)

# Footer
st.divider()
st.caption("Powered by AI Agents • Built with Streamlit")