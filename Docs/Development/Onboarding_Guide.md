# Onboarding Guide

Welcome to the `tldw_chatbook` project! This guide is designed to help you get up and running quickly.

## 🚀 Introduction

`tldw_chatbook` is a Terminal User Interface (TUI) application built with [Textual](https://textual.textualize.io/) for interacting with Large Language Models (LLMs). It allows users to chat with characters, manage conversations, and use various tools directly from the terminal.

## 🛠️ Prerequisites

- **Python 3.11+**: Ensure you have a compatible Python version installed.
- **Git**: For version control.
- **Virtual Environment**: Recommended to isolate dependencies.

## 📥 Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/rmusser01/tldw_chatbook.git
    cd tldw_chatbook
    ```

2.  **Create and activate a virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install the package in editable mode**:
    ```bash
    pip install -e .
    ```

## 🏃‍♂️ Running the Application

To start the application, simply run:

```bash
tldw-cli
```

Or via the python module:

```bash
python -m tldw_chatbook.app
```

## 🏗️ Project Architecture

The project follows a modular architecture:

- **`tldw_chatbook/app.py`**: The main entry point and application class (`TldwCli`).
- **`tldw_chatbook/css/`**: Styling files (Textual CSS).
    - `tldw_cli_modular.tcss`: The generated main stylesheet.
    - `core/`, `layout/`, `components/`: Source modules for CSS.
- **`tldw_chatbook/Widgets/`**: Reusable UI components.
- **`tldw_chatbook/UI/Screens/`**: Full-screen views.
- **`Docs/`**: Documentation.

For a more detailed deep dive, please refer to the [Developer Guide](Developer_Guide.md).

## 🧩 Key Concepts

- **Textual Framework**: The app is built on Textual. Familiarize yourself with [Widgets](https://textual.textualize.io/widgets/), [Screens](https://textual.textualize.io/guide/screens/), and [CSS](https://textual.textualize.io/guide/CSS/).
- **Reactive State**: We use Textual's `reactive` attributes for state management.
- **Event-Driven**: The app relies heavily on message passing and event handlers.

## 🤝 Contribution Guidelines

1.  **Code Style**: Follow PEP 8.
2.  **Refactoring**: We are currently undergoing a major refactoring. Please check [textual-refactoring-plan.md](textual-refactoring-plan.md) before making significant UI changes.
    - **Critical**: Do not use `outline: none` for focus states. Accessibility is a priority.
    - **Navigation**: We use a **Screen-based navigation** system. Do not introduce tab-based navigation patterns for main views.
3.  **Pull Requests**: Submit PRs with clear descriptions of changes.

## 🆘 Getting Help

If you get stuck, check the `Docs/` directory or reach out to the team.
