from .agent import agent


def main():
    input_text = input("Enter your question: ")
    result = agent.run_sync(input_text)
    print(result.output)


if __name__ == "__main__":
    main()
