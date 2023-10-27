import json
import socket
import requests

# test
def get_ip_address() -> str:
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    return ip_address


def query_llm(user_prompt):
    """Calls the API using the given user prompt."""
    # response = requests.post("http://10.0.135.48:5110/api/prompt_route", data={"user_prompt": user_prompt})
    response = requests.post(
        f"http://{get_ip_address()}:5110/api/prompt_route",
        data={"user_prompt": user_prompt},
    )
    return response


def pretty_print_json(json_data):
    """Pretty prints the given JSON data to the screen."""
    print(json.dumps(json_data, indent=4, sort_keys=True))


def main():
    last_result = None
    while True:
        user_prompt = input(
            "\n\nAsk me anything about the DP (d for details, r for reload docs, q to quit):\n"
        )
        if user_prompt == "q":
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
            last_result = response.json()
            if last_result["Sources"]:
                print(f"\nAnswer:\n{last_result['Answer']}")
            else:
                print(
                    f"\nAnswer:\nBased on our documentation, I could not find a relevant answer"
                )


if __name__ == "__main__":
    main()
