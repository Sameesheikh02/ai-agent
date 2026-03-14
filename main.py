import argparse
import os
import sys

from dotenv import load_dotenv
from google import genai
from google.genai import types

from call_function import available_functions, call_function
from prompts import system_prompt

MAX_ITERATIONS = 20  # safe upper bound for the agent's reasoning loop


def main():
    parser = argparse.ArgumentParser(description="AI Code Assistant")
    parser.add_argument("user_prompt", type=str, help="Prompt to send to Gemini")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--max-iters",
        type=int,
        default=MAX_ITERATIONS,
        help=f"Maximum agent loop iterations (default {MAX_ITERATIONS})",
    )
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")

    client = genai.Client(api_key=api_key)

    # initial message list: user's prompt as the first message
    messages = [types.Content(role="user", parts=[types.Part(text=args.user_prompt)])]

    if args.verbose:
        print(f"User prompt: {args.user_prompt}\n")

    try:
        run_agent_loop(client, messages, args.max_iters, args.verbose)
    except Exception as e:
        print("Agent failed:", e)
        raise


def run_agent_loop(client, messages, max_iters, verbose):
    """
    Repeatedly call the model, process function calls, and feed responses back into the conversation
    until the model returns a final non-function-calling response or max_iters is reached.
    """

    for iteration in range(1, max_iters + 1):
        if verbose:
            print(f"\n=== Iteration {iteration} ===")

        # Call the model once with the full conversation history (messages).
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=messages,
            config=types.GenerateContentConfig(
                tools=[available_functions], system_instruction=system_prompt
            ),
        )

        # Basic sanity checks
        if not response.usage_metadata:
            raise RuntimeError("Gemini API response appears to be malformed")

        if verbose:
            print("Prompt tokens:", response.usage_metadata.prompt_token_count)
            print("Response tokens:", response.usage_metadata.candidates_token_count)

        # 1) Append all candidates' content (if any) to the conversation so the model can see them next round.
        #    Some SDKs return response.candidates where each candidate has a .content or .parts property.
        if getattr(response, "candidates", None):
            for cand in response.candidates:
                # Prefer candidate.content if present (it will typically be a types.Content)
                if hasattr(cand, "content") and cand.content:
                    messages.append(cand.content)
                    if verbose:
                        # try to print human-friendly text if available
                        try:
                            print("Candidate appended to messages:", cand.content)
                        except Exception:
                            pass
                else:
                    # fallback: some candidate representations expose .text
                    if hasattr(cand, "text") and cand.text:
                        messages.append(types.Content(role="assistant", parts=[types.Part(text=cand.text)]))
                        if verbose:
                            print("Candidate text appended to messages:", cand.text)

        # If the model did not request any function calls, it's a final reply -> print and finish.
        if not response.function_calls:
            # Print the model's text if available; otherwise try to show candidates or response.text
            if getattr(response, "text", None):
                print("Final response:")
                print(response.text)
            else:
                # As a fallback, try candidates content or just inform the user.
                if getattr(response, "candidates", None) and len(response.candidates) > 0:
                    # Print readable candidate(s)
                    for cand in response.candidates:
                        if hasattr(cand, "content") and cand.content:
                            # try to extract text from parts if present
                            try:
                                # Some content objects have parts with text; join them
                                parts_text = []
                                for p in cand.content.parts:
                                    if getattr(p, "text", None):
                                        parts_text.append(p.text)
                                print("Final candidate content:")
                                print("\n".join(parts_text))
                            except Exception:
                                print("Final candidate (raw):", cand.content)
                        elif hasattr(cand, "text"):
                            print("Final candidate text:", cand.text)
                else:
                    print("Final response (no function calls), but no textual content found.")
            return  # finished successfully

        # If we reach here, the model did request function calls. Execute them and collect results.
        function_responses_parts = []  # list of types.Part objects (tool outputs) to send back to the model

        for function_call in response.function_calls:
            if verbose:
                print(f" - Calling function: {function_call.name}({function_call.args})")
            result = call_function(function_call, verbose)

            # Validate result structure: it must contain parts and function_response and response
            if (
                not getattr(result, "parts", None)
                or not getattr(result.parts[0], "function_response", None)
                or not getattr(result.parts[0].function_response, "response", None)
            ):
                raise RuntimeError(f"Empty function response for {function_call.name}")

            # result.parts[0] is a types.Part (the tool response part) — append it to the list
            function_responses_parts.append(result.parts[0])

            if verbose:
                try:
                    print("->", result.parts[0].function_response.response)
                except Exception:
                    print("-> (tool response appended)")

        # After running all function calls in this iteration,
        # append the list of tool responses to messages so the model sees them next loop.
        if function_responses_parts:
            # Wrap the collected parts in a Content with role="user" per the assignment instructions
            messages.append(types.Content(role="user", parts=function_responses_parts))
            if verbose:
                print(f"Appended {len(function_responses_parts)} function response(s) to conversation history.")

    # If we exit the for loop, max iterations were reached without a final response
    print(f"Max iterations ({max_iters}) reached without the model producing a final response.")
    sys.exit(1)


if __name__ == "__main__":
    main()