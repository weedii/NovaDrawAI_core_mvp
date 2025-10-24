from openai import OpenAI
import os
import requests
import re
import base64
from datetime import datetime
from pathlib import Path


# Global cost tracking
cost_tracker = []


def track_api_usage(request_type, model, usage_data, description=""):
    """
    Track API usage and costs for each request using data from the API response.
    """

    cost_info = {
        "request_type": request_type,
        "model": model,
        "description": description,
        "usage": usage_data,
    }

    cost_tracker.append(cost_info)

    # Return total cost if available in usage data
    return getattr(usage_data, "total_cost", 0) or 0


def print_cost_summary():
    """
    Print a detailed cost summary of all API requests made during the session.
    """

    if not cost_tracker:
        print("\n📊 No API requests tracked.")
        return

    print("\n" + "=" * 80)
    print("📊 API USAGE & COST SUMMARY")
    print("=" * 80)

    total_session_cost = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0

    for i, request in enumerate(cost_tracker, 1):
        print(f"\n🔸 Request #{i}: {request['request_type']}")
        print(f"   Description: {request['description']}")
        print(f"   Model: {request['model']}")

        usage = request["usage"]

        # Handle different usage field names between ChatCompletion and Response APIs
        if hasattr(usage, "prompt_tokens"):  # ChatCompletion (GPT-4o-mini)
            input_tokens = usage.prompt_tokens
            output_tokens = usage.completion_tokens
            tokens = usage.total_tokens
            print(f"   Usage data:")
            print(f"     prompt_tokens: {input_tokens}")
            print(f"     completion_tokens: {output_tokens}")
            print(f"     total_tokens: {tokens}")
            if hasattr(usage, "completion_tokens_details"):
                print(
                    f"     completion_tokens_details: {usage.completion_tokens_details}"
                )
            if hasattr(usage, "prompt_tokens_details"):
                print(f"     prompt_tokens_details: {usage.prompt_tokens_details}")
        elif hasattr(usage, "input_tokens"):  # Response (GPT-5)
            input_tokens = usage.input_tokens
            output_tokens = usage.output_tokens
            tokens = usage.total_tokens
            print(f"   Usage data:")
            print(f"     input_tokens: {input_tokens}")
            print(f"     output_tokens: {output_tokens}")
            print(f"     total_tokens: {tokens}")
            if hasattr(usage, "input_tokens_details"):
                print(f"     input_tokens_details: {usage.input_tokens_details}")
            if hasattr(usage, "output_tokens_details"):
                print(f"     output_tokens_details: {usage.output_tokens_details}")
        else:
            # Fallback: print all available usage data
            print(f"   Usage data:")
            for key, value in vars(usage).items():
                if not key.startswith("_"):
                    print(f"     {key}: {value}")
            input_tokens = getattr(
                usage, "input_tokens", getattr(usage, "prompt_tokens", 0)
            )
            output_tokens = getattr(
                usage, "output_tokens", getattr(usage, "completion_tokens", 0)
            )
            tokens = getattr(usage, "total_tokens", 0)

        # Accumulate totals
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        total_tokens += tokens

        # Try to get total cost if available
        total_cost = getattr(usage, "total_cost", None)
        if total_cost:
            total_session_cost += total_cost

    print("\n" + "-" * 80)
    print(f"📊 SESSION TOTALS:")
    print(f"   Total input tokens: {total_input_tokens:,}")
    print(f"   Total output tokens: {total_output_tokens:,}")
    print(f"   Total tokens: {total_tokens:,}")
    if total_session_cost > 0:
        print(f"💰 TOTAL SESSION COST: ${total_session_cost:.6f}")
    else:
        print("💰 TOTAL SESSION COST: Not available in API response")
    print(f"📈 Total requests made: {len(cost_tracker)}")
    print("=" * 80)


# Load environment variables from .env file
def load_env_file():
    """Manually load .env file if python-dotenv is not available"""
    env_path = Path(".env")
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    # Remove quotes if present
                    value = value.strip("\"'")
                    os.environ[key] = value


try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv not installed. Loading .env file manually...")
    load_env_file()

# Load API key from environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError(
        "OPENAI_API_KEY environment variable is not set. Please set it in your .env file or environment."
    )

client = OpenAI(api_key=api_key)


def sanitize_filename(name):
    """
    Sanitize a string to be safe for use as a filename.
    """

    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", name)
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(" .")
    # Limit length
    sanitized = sanitized[:50] if len(sanitized) > 50 else sanitized
    # Ensure it's not empty
    return sanitized if sanitized else "drawing"


def generate_drawing_steps(subject):
    """
    Ask GPT to create 5 to 7 simple drawing steps for a given subject.
    """

    prompt = f"""
    Create 7 simple kid-friendly drawing steps to draw a {subject}.
    Each step should describe what to draw next in one short sentence, like teaching a 6-year-old.
    
    IMPORTANT: Return ONLY the steps, one per line, without any numbering, introduction, or conclusion.
    Do NOT include phrases like "Here are the steps" or "Have fun drawing".
    
    Example format:
    Start with a big oval for the body
    Add a circle on top for the head
    Draw two eyes inside the head
    
    Subject: {subject}
    Steps:
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}]
        )

        if not response.choices or not response.choices[0].message.content:
            raise ValueError("Empty response from OpenAI API")

        # Track API usage
        track_api_usage(
            "Text Generation",
            "gpt-4o-mini",
            response.usage,
            f"Generate drawing steps for {subject}",
        )

        steps_text = response.choices[0].message.content.strip()

        # Split by lines and clean up
        steps = [line.strip() for line in steps_text.split("\n") if line.strip()]

        # Filter out any remaining numbered lines or intro/outro text
        clean_steps = []
        for step in steps:
            # Skip lines that look like introductions or conclusions
            if any(
                phrase in step.lower()
                for phrase in [
                    "here are",
                    "steps to",
                    "have fun",
                    "enjoy",
                    "let's",
                    "now you",
                ]
            ):
                continue
            # Remove numbering if present
            import re

            clean_step = re.sub(r"^\d+\.\s*", "", step)
            if clean_step and len(clean_step) > 10:  # Ensure it's a meaningful step
                clean_steps.append(clean_step)

        if not clean_steps:
            raise ValueError("No valid steps generated")

        return clean_steps

    except Exception as e:
        print(f"❌ Error generating drawing steps: {e}")
        raise


def create_session_folder(subject):
    """
    Create a unique session folder for this drawing session.
    """

    try:
        sanitized_subject = sanitize_filename(subject)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_name = f"{sanitized_subject}_{timestamp}"
        session_path = Path("steps") / session_name
        session_path.mkdir(parents=True, exist_ok=True)
        return session_path
    except Exception as e:
        print(f"❌ Error creating session folder: {e}")
        raise


def download_and_save_image(image_url, file_path):
    """
    Download an image from URL and save it to the specified path.
    """

    try:
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()

        with open(file_path, "wb") as f:
            f.write(response.content)

        print(f"💾 Image saved: {file_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to save image: {e}")
        return False


def generate_step_image(
    step_description,
    subject,
    step_number,
    session_folder,
    all_steps=None,
    previous_image_path=None,
):
    """
    Generate a black-and-white kid-friendly image for each step using the Responses API with gpt-5.
    Uses base64 image input for multi-turn editing.
    """

    # Validate session_folder is a Path object
    if not isinstance(session_folder, Path):
        session_folder = Path(session_folder)

    try:
        # Create enhanced prompt with context
        image_prompt = (
            f"We are creating a step-by-step drawing tutorial to teach kids how to draw a {subject}. "
            f"This is step {step_number} of the tutorial. "
            f"Your task: {step_description} "
            f"IMPORTANT: Only do what is described in this step. Do not add elements from future steps. "
            f"Do NOT include any text, labels, or words in the image. Only draw the shapes and lines. "
            f"Style: Simple black and white line drawing, clean cartoon style, no shading or color. "
            f"Make it easy for children to copy. Show only the basic shapes and lines needed for this specific step."
        )

        # Prepare input content
        input_content = [{"type": "input_text", "text": image_prompt}]

        # If we have a previous image, include it for editing
        if (
            previous_image_path
            and step_number > 1
            and Path(previous_image_path).exists()
        ):
            # Encode previous image as base64
            with open(previous_image_path, "rb") as f:
                previous_image_base64 = base64.b64encode(f.read()).decode("utf-8")

            # Add previous image to input
            input_content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{previous_image_base64}",
                }
            )

            # Update prompt for editing
            image_prompt = (
                f"We are creating a step-by-step drawing tutorial to teach kids how to draw a {subject}. "
                f"We are now at step {step_number} of the tutorial. "
                f"Start from the given image and add only what is described in this step: {step_description} "
                f"IMPORTANT: Keep all existing elements exactly as they are. Do not modify or remove anything from the previous steps. "
                f"Only add what this specific step describes. Do not add elements from future steps. "
                f"Do NOT include any text, labels, or words in the image. Only draw the shapes and lines. "
                f"Style: Simple black and white line drawing, clean cartoon style, no shading or color. "
                f"Make it easy for children to copy."
            )
            input_content[0]["text"] = image_prompt

        # Print the full prompt structure being sent to the API
        print(f"🔍 FULL PROMPT STRUCTURE for step {step_number}:")
        print(f"   Model: gpt-5")
        print(f"   Input structure:")
        print(f"     - Role: user")
        print(f"     - Content:")
        for idx, content_item in enumerate(input_content):
            if content_item["type"] == "input_text":
                print(f"       [{idx}] Type: input_text")
                print(f"           Text: \"{content_item['text']}\"")
            elif content_item["type"] == "input_image":
                print(f"       [{idx}] Type: input_image")
                print(f"           Image URL: data:image/png;base64,[BASE64_DATA]")
        print(f"   Tools: image_generation (quality: low)")
        print("-" * 80)

        # Use Responses API with gpt-5
        response = client.responses.create(
            model="gpt-5",
            input=[{"role": "user", "content": input_content}],
            tools=[{"type": "image_generation", "quality": "low"}],
        )

        # Track API usage
        if hasattr(response, "usage") and response.usage:
            track_api_usage(
                "Image Generation",
                "gpt-5",
                response.usage,
                f"Generate step {step_number} image for {subject}",
            )

        # Extract image generation calls from response
        image_generation_calls = [
            output
            for output in response.output
            if output.type == "image_generation_call"
        ]

        if not image_generation_calls:
            raise ValueError("No image generation calls returned from OpenAI API")

        # Get the image data
        image_call = image_generation_calls[0]
        image_base64 = image_call.result

        print(f"✅ Step {step_number} image generated successfully")

        # Save the image to the session folder
        sanitized_subject = sanitize_filename(subject)
        filename = f"step_{step_number:02d}_{sanitized_subject}.png"
        file_path = session_folder / filename

        # Decode base64 and save directly
        try:
            image_bytes = base64.b64decode(image_base64)
            with open(file_path, "wb") as f:
                f.write(image_bytes)
            print(f"💾 Image saved: {file_path}")
            return str(file_path)
        except Exception as e:
            print(f"❌ Failed to save image: {e}")
            return None

    except Exception as e:
        print(f"❌ Error generating step {step_number} image: {e}")
        raise


def main():
    try:
        # Input validation
        subject = input("What do you want to draw? (e.g., owl, fish, cat): ").strip()

        if not subject:
            print("❌ Error: Subject cannot be empty. Please enter a valid subject.")
            return

        if len(subject) > 100:
            print("❌ Error: Subject name too long. Please use a shorter name.")
            return

        print(f"\nGenerating drawing guide for: {subject}\n")

        # Create session folder for this drawing session
        session_folder = create_session_folder(subject)
        print(f"📁 Session folder created: {session_folder}\n")

        # Generate drawing steps
        steps = generate_drawing_steps(subject)
        print("📝 Steps:\n")
        for step in steps:
            print(step)

        print("\n🎨 Generating and saving images...\n")
        saved_images = []

        # Generate images with progress tracking
        for i, step in enumerate(steps, start=1):
            try:
                print(f"Generating image {i}/{len(steps)}...")

                # Get previous image path if it exists
                previous_image_path = None
                if i > 1 and saved_images:
                    previous_image_path = saved_images[
                        -1
                    ]  # Last successfully saved image

                saved_path = generate_step_image(
                    step,
                    subject,
                    i,
                    session_folder,
                    all_steps=steps,
                    previous_image_path=previous_image_path,
                )

                if saved_path is not None:
                    saved_images.append(saved_path)
                else:
                    print(f"⚠️ Warning: Failed to generate image for step {i}")

            except Exception as e:
                print(f"⚠️ Warning: Failed to generate image for step {i}: {e}")
                print("Continuing with next step...")
                continue

        if saved_images:
            print(f"\n🎉 Complete! Images saved in: {session_folder}")
            print(f"📊 Total images generated: {len(saved_images)}/{len(steps)}")
        else:
            print(f"\n❌ No images were successfully generated.")

        # Print cost summary
        print_cost_summary()

    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user.")
        print_cost_summary()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        print("Please check your API key and internet connection.")
        print_cost_summary()


if __name__ == "__main__":
    main()
