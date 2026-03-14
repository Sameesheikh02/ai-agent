# AI Code Assistant

An AI-powered coding assistant that can analyze, modify, and execute code using the Gemini API.

## Features

- Reads files in a project
- Executes Python scripts
- Writes and edits files
- Iteratively fixes bugs using an agent loop

## Tech Stack

- Python
- Gemini API
- Function calling
- Subprocess execution

## Installation

Clone the repository:

git clone https://github.com/YOUR_USERNAME/ai-agent.git

Move into the project:

cd ai-agent

Install dependencies:

pip install -r requirements.txt

Create a `.env` file:

GEMINI_API_KEY=your_api_key_here

Run the assistant:

python main.py "Fix the bug in calculator.py"

## Example

python main.py "List all files in the project"

## Disclaimer

Make sure your `.env` file contains a valid API key.
