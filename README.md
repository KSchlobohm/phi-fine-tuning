# Learning sample
This is a sample project to learn about Python, LLMs, and other technologies.

## Getting started with Python

1. Pip only runs in a virtual environment
    ```sh
    export PIP_REQUIRE_VIRTUALENV true
    ```

    ```ps1
    setx PIP_REQUIRE_VIRTUALENV true
    ```

1. *Create a Virtual Environment*
    ```sh
    python -m venv ./venv
    ```

1. *Freeze Requirements*
    ```sh
    pip freeze > requirements.txt
    ```

1. *Install Requirements*
    ```sh
    pip install -r requirements.txt
    ```

## Prerequisites
- [startup.md](startup.md)