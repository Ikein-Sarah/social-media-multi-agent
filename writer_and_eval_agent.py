import asyncio
import yt_dlp
from agents import Agent, Runner, WebSearchTool, function_tool
from openai import OpenAI
import os
from dotenv import load_dotenv
from dataclasses import dataclass

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


async def main():
    # Get TikTok URL from user
    tiktok_url = input("Please enter the TikTok URL: ").strip()

    # Download and transcribe TikTok
    print("Downloading and transcribing TikTok...")
    transcript = transcribe_tiktok(tiktok_url)
    msg = f"Generate a LinkedIn post based on this TikTok transcript: {transcript}"

    input_items = [{"content": msg, "role": "user"}]
    latest_content: str | None = None

    # Run the content writer agent with feedback loop
    print("\n✍ Writing content with feedback...")
    max_attempts = 3
    for attempt in range(max_attempts):
        print(f"\n--- Attempt {attempt + 1}/{max_attempts} ---")

        # Generate LinkedIn post
        generated_content_result = await Runner.run(
            content_writer_agent,
            input_items,
        )
        latest_content = generated_content_result.final_output
        print("Content generated:", latest_content)

        # Evaluate the generated post
        evaluator_input = [{"content": f"Generated LinkedIn post: {latest_content}", "role": "user"}]
        evaluator_result = await Runner.run(evaluator, evaluator_input)
        result: EvaluationFeedback = evaluator_result.final_output

        print(f"Evaluator score: {result.score}/10")
        print(f"Evaluator feedback: {result.feedback}")

        # If score is good, break out of the loop
        if result.score >= 8:
            print("Score is satisfactory!")
            break

        # Add feedback for the next iteration
        input_items.append({"content": f"Feedback: {result.feedback}", "role": "user"})

    else:
        print("\nMaximum feedback iterations reached. Using best available content.")

    print(f"\n🎉 Final LinkedIn Post:\n{latest_content}")


if __name__ == "__main__":
    asyncio.run(main())