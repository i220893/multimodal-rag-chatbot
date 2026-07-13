import subprocess

def call_ollama(prompt, model="mistral:7b-instruct-q4_K_M", temperature=0.5):
    """
    Calls local Ollama model on Windows.
    Temperature is passed inside the prompt (Ollama recommended way).
    """

    # Encode temperature into prompt metadata
    system_prompt = f"<s>[INST] <<SYS>> Temperature={temperature}. You are a helpful assistant. <</SYS>>\n{prompt} [/INST]"

    command = [
        "ollama",
        "run",
        model,
    ]

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    output, error = process.communicate(system_prompt)

    if error and "error" in error.lower():
        print("LLM ERROR:", error)

    return output.strip()
