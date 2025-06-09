from transformers import AutoTokenizer

def main():
    # Load the Qwen3 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)

    # Test strings
    test_strings = [
        "</fork>",
        "\n<think>\n\n</think>\n\n<fork budget=100>[\"(44 + 19) - 35\", \"44 + 19 + 35\", \"44 * 19 - 35\", \"44 - (19 + 35)\"]</fork><response>[{\"child task\": \"(44 + 19) - 35\", \"child response\": null}, {\"child task\": \"44 + 19 + 35\", \"child response\": null}, {\"child task\": \"44 * 19 - 35\", \"child response\": null}, {\"child task\": \"44 - (19 + 35)\", \"child response\": null}]</response>"
    ]

    # Tokenize both strings and compare the tokenization of </fork>
    ids_0 = tokenizer.encode(test_strings[0], add_special_tokens=False)
    ids_1 = tokenizer.encode(test_strings[1], add_special_tokens=False)

    # Find the position of </fork> in the second string
    fork_str = "</fork>"
    fork_start = test_strings[1].find(fork_str)
    fork_end = fork_start + len(fork_str)

    # Tokenize the substring corresponding to </fork> in the second string
    ids_1_fork = tokenizer.encode(test_strings[1][fork_start:fork_end], add_special_tokens=False)

    print(f"Token IDs for standalone </fork>: {ids_0}")
    print(f"Token IDs for </fork> in context: {ids_1_fork}")

    if ids_0 == ids_1_fork:
        print("The </fork> tokenization is IDENTICAL in both cases.")
    else:
        print("The </fork> tokenization DIFFERS between the two cases.")

    # For completeness, print the tokens as well
    print("Tokens for standalone </fork>:", tokenizer.convert_ids_to_tokens(ids_0))
    print("Tokens for </fork> in context:", tokenizer.convert_ids_to_tokens(ids_1_fork))

if __name__ == "__main__":
    main()
