import json
import time
import requests

def get_ip_address() -> str:
#    hostname = socket.gethostname()
#    ip_address = socket.gethostbyname(hostname)
#    return ip_address
    return "127.0.0.1"


def query_llm(user_prompt: str):
    """Calls the API using the given user prompt."""
    response = None
    for _ in range(3):
        try:
            response = requests.post(
                f"http://{get_ip_address()}:5110/api/prompt_route",
                data={"user_prompt": user_prompt},
            )
            if response.status_code == 200:
                break
            else:
                print(f"API call failed with status code {response.status_code}. Retrying in 4s...")
                time.sleep(4)
        except requests.exceptions.RequestException:
            print("API call failed due to a network error. Retrying in 4s...")
            time.sleep(4)

    return response


def pretty_print_json(json_data):
    """Pretty prints the given JSON data to the screen."""
    print(json.dumps(json_data, indent=4, sort_keys=True))


def main():
    last_result = None
    while True:
        try:
            user_prompt = input(
                "\n\nAsk me anything about the Pubengine (d for details, p for download & process docs, q to quit):\n"
            )
            if user_prompt == "":
                continue
            if user_prompt == "q":
                print("Closing..")
                break
            elif user_prompt == "d":
                print("\nDetails:")
                pretty_print_json(last_result)
            elif user_prompt == "p":
                # response = requests.get(http://10.0.135.48:5110/api/run_ingest)
                print("Downloading documententation from S3... ", end="")
                response = requests.get(f"http://{get_ip_address()}:5110/api/sync_s3_to_source_docs")
                print("DONE")
                print("Reloading/ingesting... ", end="")
                response = requests.get(f"http://{get_ip_address()}:5110/api/run_ingest")
                print("DONE")
                print("--- PLEASE RESTART THE SERVER ---")
            else:
                response = query_llm(user_prompt)
                if response is None:
                    print("Connection to localGPT times out... Try again")
                    continue
                last_result = response.json()
                if last_result["Sources"]:
                    print(f"\nAnswer:\n{last_result['Answer']}\n")
                    print(f"References page{'s' if len(last_result['Sources']) > 1 else ''} with more details and examples regarding this answer:")
                    for source in last_result["Sources"]:
                        source_page_id = source[0]
                        clean_page_id = source_page_id[:source_page_id.find("_")]
                        endpoint = f"https://globalitconfluence.us.aegon.com/pages/viewpage.action?pageId={clean_page_id}"
                        print(f" - {endpoint}")
                else:
                    print(
                        "\nAnswer:\nBased on our documentation, I could not find a relevant answer. Please try again with a different or more elaborate question."
                    )
        except KeyboardInterrupt:
            print("\nCTRL+C detected. Use CTRL+Insert to copy..")
            try:
                user_prompt = input("\nDo you want to quit (q)?:\n")
                if user_prompt.lower() == "q":
                    print("Closing..")
                    break
                continue
            except KeyboardInterrupt:
                print("Closing..")
                break

if __name__ == "__main__":
    main()

