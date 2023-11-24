import json
import time
import requests
import os

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
                print(f"API call failed with status code {response.status_code}. Retrying...")
                time.sleep(2)
        except requests.exceptions.RequestException:
            print("API call failed due to a network error. Retrying...")
            time.sleep(2)

    return response


def pretty_print_json(json_data):
    """Pretty prints the given JSON data to the screen."""
    print(json.dumps(json_data, indent=4, sort_keys=True))

def main():
    last_result = None
    while True:
        try:
            user_prompt = input(
                "\n\nAsk me anything about the DP (d for details, r for reload docs, q to quit):\n"
            )
            if user_prompt == "":
                continue
            if user_prompt == "q":
                print("Closing..")
                break
            elif user_prompt == "d":
                print("\nDetails:")
                pretty_print_json(last_result)
            elif user_prompt == "r":
                print("Reloading/ingesting")
                # response = requests.get("http://10.0.135.48:5110/api/run_ingest")
                response = requests.get(f"http://{get_ip_address()}:5110/api/run_ingest")
                print(".. done Reloading/ingesting")
            else:
                response = query_llm(user_prompt)
                if response is None:
                    print("Connection to localGPT times out... Try again")
                    continue
                last_result = response.json()
                if last_result["Sources"]:
                    print(f"\nAnswer:\n{last_result['Answer']}")
                    # filename = os.path.splitext(os.path.basename(last_result["Sources"][0][0]))[0]
                    # print(f"\nURL: {filename}")
                else:
                    print(
                        "\nAnswer:\nBased on our documentation, I could not find a relevant answer"
                    )
        except KeyboardInterrupt:
            print("\nCTRL+C detected. Use CTRL+Insert to copy..")
            user_prompt = input(
                "\nDo you want to continue (y/n)?:\n"
            )
            try:
                if user_prompt != "y":
                    print("Closing..")
                    break
                else:
                    continue
            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    main()
