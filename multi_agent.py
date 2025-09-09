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



load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


# --------------------------------------------------------------
# Step 1: Define tools
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


# --------------------------------------------------------------
# Step 2: Download and transcribe function
# --------------------------------------------------------------

def download_tiktok_audio(url: str) -> str:
    """Download audio from TikTok URL using yt-dlp"""
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'tiktok_audio.%(ext)s',
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
            ydl.download([url])
        return "tiktok_audio.mp3"
    except Exception as e:
        print(f"Download error: {e}")
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
        print(f"Transcription error: {e}")
        raise


def transcribe_tiktok(url: str) -> str:
    """Main function to handle TikTok URL"""
    print(f"Downloading from: {url}")

    # Download audio
    audio_path = download_tiktok_audio(url)
    print("Download complete")

    # Transcribe
    print("🔊 Transcribing audio...")
    transcript = transcribe_audio(audio_path)
    print("Transcription complete")

    # Clean up
    if os.path.exists(audio_path):
        os.remove(audio_path)

    return transcript


# Define the evaluator agent
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


# --------------------------------------------------------------
# Step 3: Define the storage agent (NOTION)
# --------------------------------------------------------------


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
                            "content": title[:200]  # Notion has title length limits
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
            notion_link = f"https://www.notion.so/{page_id.replace('-', '')}"
            print(f"✅ Saved to Notion: {notion_link}")
            return notion_link
        else:
            print(f"❌ Notion error: {response.text}")
            # Fallback: return success message but keep content
            print("📋 Post content was generated successfully!")
            return f"Notion Error - but post is ready: {content[:100]}..."

    except Exception as e:
        print(f"❌ Notion connection failed: {e}")
        return f"Connection failed - but post is ready: {content[:100]}..."


content_storage_agent = Agent(
    name="LinkedIn Content Storage Agent",
    instructions=f"""Save LinkedIn posts to Notion with clear, descriptive titles.
    Use this naming format: 'LinkedIn Post - [Topic Summary]'
    Make sure the post is saved properly and return the Notion link.""",
    model="gpt-4o-mini",
    tools=[save_to_notion],
    output_type=StoredPosts,
)


# --------------------------------------------------------------
# Step 4: Define the content scheduling agent
# --------------------------------------------------------------

# Authenticate Google Calendar API
SERVICE_ACCOUNT_FILE = "social-media-agent.json"
SCOPES = ["https://www.googleapis.com/auth/calendar"]

def authenticate_google_calendar():
    """Authenticate and return Google Calendar service"""
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    calendar_service = build("calendar", "v3", credentials=creds)
    return calendar_service

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
    Args:
        event_title (str): The title of the event.
        event_description (str): The description of the event.
        suggested_time (str): The suggested posting time in ISO format.
    Returns:
        str: The HTML link to the created event.
    """

    suggested_time = datetime.datetime.fromisoformat(suggested_time)

    event = {
        'summary': event_title,
        'description': event_description,
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

    print(f"✅ Event created: {event_result.get('htmlLink')}")
    return event_result.get("htmlLink")


content_scheduling_agent = Agent(
    name="Social Media Content Scheduling Agent",
    instructions=f"""Schedule ONLY ONE posting event for this LinkedIn content. 
    Choose the next available weekday (Monday-Friday) in the morning (7-8 AM Lagos time).
    DO NOT create multiple events - create only one event for the optimal posting time.
    Include the generated content in the event description.
    Return only the single event link.""",
    model="gpt-4o-mini",
    tools=[get_time_now_lagos, add_to_calendar],
)


# --------------------------------------------------------------
# Step 5: Main Workflow Function
# --------------------------------------------------------------

async def main():
    """Main workflow: TikTok URL -> Transcript -> LinkedIn Post -> Evaluation -> Save -> Schedule"""

    # Get TikTok URL from user
    tiktok_url = input("Please enter the TikTok URL: ").strip()

    try:
        # Step 1: Download and transcribe TikTok
        print("\n Starting LinkedIn Content Pipeline...")
        print(f"Processing: {tiktok_url}")
        transcript = transcribe_tiktok(tiktok_url)
        print(f"✅ Transcription complete ({len(transcript)} characters)")

        # Step 2: Generate LinkedIn post with feedback loop
        msg = f"Generate a LinkedIn post based on this TikTok transcript: {transcript}"
        input_items = [{"content": msg, "role": "user"}]
        latest_content = None

        print("\n✍️ Writing LinkedIn post with feedback...")
        max_attempts = 2
        for attempt in range(max_attempts):
            print(f"\n--- Attempt {attempt + 1}/{max_attempts} ---")

            # Generate LinkedIn post
            generated_content_result = await Runner.run(
                content_writer_agent,
                input_items,
            )
            latest_content = generated_content_result.final_output
            print("Content generated:", latest_content[:100] + "..." if len(latest_content) > 100 else latest_content)

            # Evaluate the generated post
            evaluator_input = [{"content": f"Generated LinkedIn post: {latest_content}", "role": "user"}]
            evaluator_result = await Runner.run(evaluator, evaluator_input)
            result: EvaluationFeedback = evaluator_result.final_output

            print(f"Evaluator score: {result.score}/10")

            # If score is good, break out of the loop
            if result.score >= 8:
                print("✅ Post meets quality standards!")
                break

            # Add feedback for the next iteration
            input_items.append({"content": f"Feedback: {result.feedback}", "role": "user"})

        else:
            print("\nMaximum feedback iterations reached. Using best available content.")

        # Step 3: Save to Google Docs
        print("\n💾 Storing content...")
        storage_result = await Runner.run(
            content_storage_agent,
            input=f"Save this LinkedIn post: {latest_content}",
        )


        # Step 4: Schedule the post
        print("\n Scheduling post...")
        schedule_result = await Runner.run(
            content_scheduling_agent,
            input=f"Schedule this LinkedIn post for posting: {latest_content}",
        )
        print("✅ Scheduled on calendar!")

        # Final output
        print(f"\n Pipeline completed successfully!")
        print(f" Final post: {len(latest_content)} characters")
        print(f" Calendar: Scheduled successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
