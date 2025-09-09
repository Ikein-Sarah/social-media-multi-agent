import asyncio
import yt_dlp
from agents import Agent, Runner, WebSearchTool, function_tool
from openai import OpenAI
import os
from dotenv import load_dotenv

# --------------------------------------------------------------
# Step 1: Setup
# --------------------------------------------------------------
load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


# --------------------------------------------------------------
# Step 2: Define tools
# --------------------------------------------------------------

@function_tool
def generate_linkedin_post(transcript: str) -> str:
    """Generate LinkedIn post from transcript"""
    prompt = f"""
    Rewrite this TikTok content into a professional LinkedIn post:
    - Engaging but professional
    - Short paragraphs
    - Clear call to action

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
                          that fits LinkedIn’s audience. Avoid slang, keep sentences concise, 
                          and highlight insights or value the audience can take away. 
                          You may search the web for up-to-date information on the topic and fill in some useful details if needed"""),
    tools=[WebSearchTool(), generate_linkedin_post],
    output_type=str,
)


# --------------------------------------------------------------
# Step 3: Download and transcribe function
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
    print(f" Downloading from: {url}")

    # Download audio
    audio_path = download_tiktok_audio(url)
    print("✅ Download complete")

    # Transcribe
    print("🔊 Transcribing audio...")
    transcript = transcribe_audio(audio_path)
    print("✅ Transcription complete")

    # Clean up
    if os.path.exists(audio_path):
        os.remove(audio_path)

    return transcript


# --------------------------------------------------------------
# Step 4: Main execution
# --------------------------------------------------------------

async def main():
    # Input URL directly
    url = "https://www.tiktok.com/@ikein_sarah/video/7457292038384405765?is_from_webapp=1&sender_device=pc&web_id=7443933951452923447"

    try:
        # Get transcript
        transcript = transcribe_tiktok(url)
        print(f" Transcript: {transcript[:200]}...")  # Show first 200 chars

        # Generate LinkedIn post
        msg = f"Generate a LinkedIn post based on this video transcript: {transcript}"
        input_items = [{"content": msg, "role": "user"}]

        print(" Generating LinkedIn post...")
        result = await Runner.run(content_writer_agent, input_items)

        print("\n FINAL LINKEDIN POST:")
        print("=" * 50)
        print(result.final_output)
        print("=" * 50)

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())