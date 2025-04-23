# Contributing to SocialSynth-AI

Thank you for considering contributing to SocialSynth-AI! This guide outlines the process for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by the following principles:

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the best possible outcome for the project's users
- Show empathy towards other community members

## How Can I Contribute?

### Reporting Bugs

- Check if the bug has already been reported in the Issues section
- Use the bug report template if available
- Include detailed steps to reproduce the issue
- Describe the expected behavior and what actually happened
- Include screenshots if applicable
- Specify the version of the software and environment

### Suggesting Features

- Check if the feature has already been suggested
- Describe the feature in detail
- Explain why it would be valuable to the project
- Consider how it fits into the existing architecture

### Pull Requests

1. Fork the repository
2. Create a new branch from `main`
3. Make your changes
4. Run the tests to ensure they pass
5. Update documentation if necessary
6. Submit a pull request that describes what you've done

## Development Environment Setup

1. Clone the repository
   ```bash
   git clone https://github.com/razaabbasnextgen/SocialSynth-AI.git
   cd SocialSynth-AI
   ```

2. Create a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install development dependencies
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

4. Set up pre-commit hooks (optional)
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Coding Standards

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Use type hints where appropriate
- Write docstrings for all public functions, classes, and methods
- Keep lines to a maximum of 88 characters
- Run `black` and `isort` to format your code
- Run `mypy` for type checking

## Testing

- Write tests for all new functionality
- Ensure all tests pass before submitting a pull request
- Run the test suite with `pytest`

## Documentation

- Update documentation for any changes in functionality
- Follow Google-style docstrings
- Include examples where appropriate

## Commit Messages

- Use clear, descriptive commit messages
- Begin with a short (50 chars or less) summary
- Follow with a more detailed explanation if necessary
- Reference issues and pull requests where appropriate

Thank you for contributing to SocialSynth-AI! 