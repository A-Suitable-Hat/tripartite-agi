# Installation Guide

This guide provides detailed installation instructions for the Tripartite AGI Architecture system.

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation Methods](#installation-methods)
  - [Quick Install (Mock LLM)](#quick-install-mock-llm)
  - [Full Install (Real LLM)](#full-install-real-llm)
- [Verification](#verification)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements

- **Python:** 3.8 or higher (3.9+ recommended)
- **Operating System:** Windows, macOS, or Linux
- **RAM:** 512 MB minimum
- **Disk Space:** 100 MB

### Recommended for Real LLM Use

- **Python:** 3.10 or 3.11
- **RAM:** 2 GB or more
- **Internet:** Stable connection for API calls
- **Anthropic API Key:** Required for real LLM usage

### Checking Your Python Version

```bash
python --version
# or
python3 --version
```

If Python is not installed, download from [python.org](https://www.python.org/downloads/).

## Installation Methods

### Quick Install (Mock LLM)

This method uses a mock LLM client - perfect for testing, learning, and development without API costs.

#### Step 1: Clone or Download Repository

**Option A: Using Git**
```bash
git clone https://github.com/A-Suitable-Hat/tripartite-agi.git
cd tripartite-agi
```

**Option B: Download ZIP**
1. Download the repository as a ZIP file
2. Extract to desired location
3. Navigate to the directory in terminal

#### Step 2: Install Dependencies

```bash
# On Unix/Mac/Linux:
pip3 install numpy

# On Windows:
pip install numpy
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

Note: The anthropic package will be skipped if not needed for mock mode.

#### Step 3: Verify Installation

```bash
python tripartite_agi_complete.py --test
```

Expected output:
```
✓ System Integrity: 4/4 passed
✓ Undeliberables: 5/5 passed
✓ Veto Mechanism: 2/2 passed
...
ALL TESTS PASSED (40/40)
```

#### Step 4: Run Demo

```bash
python tripartite_agi_complete.py
```

Or try the basic example:

```bash
python examples/basic_usage.py
```

### Full Install (Real LLM)

This method enables actual LLM reasoning using Anthropic's Claude API.

#### Step 1: Install Core Dependencies

Follow Steps 1-2 from Quick Install above.

#### Step 2: Install Anthropic SDK

```bash
pip install anthropic
```

Verify installation:
```bash
python -c "import anthropic; print(anthropic.__version__)"
```

#### Step 3: Obtain Anthropic API Key

1. Sign up at [Anthropic Console](https://console.anthropic.com/)
2. Navigate to API Keys section
3. Create a new API key
4. Copy the key (starts with `sk-ant-...`)

⚠️ **Security Warning:** Never commit API keys to version control!

#### Step 4: Set API Key Environment Variable

**On Unix/Mac/Linux:**
```bash
# Temporary (current session only)
export ANTHROPIC_API_KEY='sk-ant-your-key-here'

# Permanent (add to ~/.bashrc or ~/.zshrc)
echo 'export ANTHROPIC_API_KEY="sk-ant-your-key-here"' >> ~/.bashrc
source ~/.bashrc
```

**On Windows CMD:**
```cmd
# Temporary
set ANTHROPIC_API_KEY=sk-ant-your-key-here

# Permanent
setx ANTHROPIC_API_KEY "sk-ant-your-key-here"
```

**On Windows PowerShell:**
```powershell
# Temporary
$env:ANTHROPIC_API_KEY='sk-ant-your-key-here'

# Permanent
[System.Environment]::SetEnvironmentVariable('ANTHROPIC_API_KEY','sk-ant-your-key-here','User')
```

Verify it's set:
```bash
# Unix/Mac/Linux
echo $ANTHROPIC_API_KEY

# Windows CMD
echo %ANTHROPIC_API_KEY%

# Windows PowerShell
echo $env:ANTHROPIC_API_KEY
```

#### Step 5: Enable Real LLM in Code

Edit `tripartite_agi_complete.py`:

**Line 308-318** - Uncomment the API call implementation:

```python
# BEFORE (commented out):
# try:
#     response = self.client.messages.create(
#         model=self.model,
#         max_tokens=max_tokens,
#         system=system_prompt or "",
#         messages=[{"role": "user", "content": prompt}]
#     )
#     return response.content[0].text
# except Exception as e:
#     return f"API_ERROR: {str(e)}"

# AFTER (uncommented):
try:
    response = self.client.messages.create(
        model=self.model,
        max_tokens=max_tokens,
        system=system_prompt or "",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text
except Exception as e:
    return f"API_ERROR: {str(e)}"
```

And comment out or remove the `raise NotImplementedError(...)` block below it.

**Line 293** - Uncomment the Anthropic client import:

```python
# BEFORE:
# import anthropic  # Uncomment this

# AFTER:
import anthropic
```

**Line 295** - Uncomment the client initialization:

```python
# BEFORE:
# self.client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var

# AFTER:
self.client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var
```

#### Step 6: Verify Real LLM Setup

Run the real LLM example:

```bash
python examples/real_llm_setup.py
```

If configured correctly, you should see:
```
✓ System created successfully with real LLM
Processing sensor data...
(This will make 6 API calls to Claude - may take 2-5 seconds)
```

## Verification

### Running Tests

```bash
# Run all automated tests
python tripartite_agi_complete.py --test
```

Expected output: `ALL TESTS PASSED (40/40)`

### Running Examples

```bash
# Basic examples (mock LLM)
python examples/basic_usage.py

# Real LLM example (requires API key)
python examples/real_llm_setup.py
```

### Checking System Status

```python
from tripartite_agi_complete import create_system

agi = create_system()
agi.print_status()
```

## Configuration

### Choosing LLM Model

In `tripartite_agi_complete.py`, line 291:

```python
def __init__(self, model: str = "claude-sonnet-4-20250514"):
```

Available models:
- `claude-sonnet-4-20250514` (recommended, balanced)
- `claude-opus-4-20250514` (highest quality, slower, more expensive)
- `claude-haiku-4-20250514` (fastest, cheapest, lower quality)

### Adjusting Token Limits

In `tripartite_agi_complete.py`, each Aspect's query uses 256 tokens by default:

```python
# Line ~3691 in Aspect.generate_proposal()
response = self.llm.query(
    prompt=prompt,
    system_prompt=self.system_prompt,
    max_tokens=256  # Adjust this value
)
```

Higher values = more detailed responses but slower and more expensive.

### History Size

Control how many past incidents are stored:

```python
# When creating system
from tripartite_agi_complete import SubconsciousLayer

subconscious = SubconsciousLayer(max_history=50)  # Default is 50
```

Larger values = more memory usage but better history-based reasoning.

## Troubleshooting

### Common Issues

#### Issue: `ModuleNotFoundError: No module named 'numpy'`

**Solution:**
```bash
pip install numpy
```

#### Issue: `ModuleNotFoundError: No module named 'anthropic'`

**Solution:**
```bash
pip install anthropic
```

#### Issue: `API_ERROR: 401 Unauthorized`

**Causes:**
- API key not set
- Invalid API key
- API key expired

**Solution:**
```bash
# Verify key is set
echo $ANTHROPIC_API_KEY  # Unix/Mac
echo %ANTHROPIC_API_KEY%  # Windows

# Re-set the key
export ANTHROPIC_API_KEY='sk-ant-your-key-here'

# Verify it's correct (should start with sk-ant-)
```

#### Issue: `NotImplementedError: AnthropicLLMClient requires API key`

**Cause:** The AnthropicLLMClient implementation is still commented out.

**Solution:** Follow Step 5 of Full Install to uncomment the implementation.

#### Issue: Tests fail with "Ontology integrity check failed"

**Cause:** The harm ontology may have been modified incorrectly.

**Solution:** Re-download the original `tripartite_agi_complete.py` file.

#### Issue: Slow performance with real LLM

**Expected Behavior:**
- Each deliberation makes 6 API calls
- Takes 2-5 seconds per cycle
- This is normal for sequential API calls

**Optimization Options:**
- Reduce max_tokens (faster responses)
- Use claude-haiku model (faster, cheaper)
- Batch scenarios rather than real-time processing

#### Issue: High API costs

**Cost Management:**
- Use mock LLM for development/testing
- Switch to real LLM only for production/demos
- Monitor usage at [console.anthropic.com](https://console.anthropic.com/)
- Set usage limits in Anthropic dashboard

**Estimated Costs (Claude Sonnet 4):**
- ~$0.01-0.03 per deliberation cycle
- ~$1-3 per 100 deliberations
- Varies based on prompt length and response tokens

### Python Version Issues

#### Using Python 3.8 or 3.9

Some type hints use features from Python 3.10+. If you encounter syntax errors:

**Option 1:** Upgrade to Python 3.10+

**Option 2:** Modify type hints:
```python
# Change this (Python 3.10+):
def foo() -> list[str]:

# To this (Python 3.8+):
from typing import List
def foo() -> List[str]:
```

### Platform-Specific Issues

#### Windows: `python` command not found

Use `python` instead of `python3`, or add Python to PATH.

#### Mac: Permission denied

Use `python3` and `pip3`:
```bash
python3 tripartite_agi_complete.py
pip3 install -r requirements.txt
```

#### Linux: Multiple Python versions

Use explicit version:
```bash
python3.10 tripartite_agi_complete.py
python3.10 -m pip install -r requirements.txt
```

## Virtual Environment (Recommended)

Using a virtual environment isolates dependencies:

### Creating Virtual Environment

```bash
# Create
python -m venv venv

# Activate (Unix/Mac)
source venv/bin/activate

# Activate (Windows CMD)
venv\Scripts\activate.bat

# Activate (Windows PowerShell)
venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Deactivate when done
deactivate
```

## Uninstallation

To remove the system:

```bash
# Remove dependencies
pip uninstall numpy anthropic

# Delete repository
rm -rf tripartite-agi  # Unix/Mac
rmdir /s tripartite-agi  # Windows

# Remove environment variable
unset ANTHROPIC_API_KEY  # Unix/Mac
set ANTHROPIC_API_KEY=  # Windows
```

## Getting Help

If you encounter issues not covered here:

1. Check [README.md](README.md) for general information
2. Review [examples/](examples/) for usage patterns
3. Read [docs/architecture.md](docs/architecture.md) for design details
4. Open an issue on GitHub with:
   - Your Python version (`python --version`)
   - Your OS and version
   - Full error message
   - Steps to reproduce

## Next Steps

After successful installation:

1. ✓ Run automated tests: `python tripartite_agi_complete.py --test`
2. ✓ Try basic examples: `python examples/basic_usage.py`
3. ✓ Read architecture docs: [docs/architecture.md](docs/architecture.md)
4. ✓ Explore the code structure
5. ✓ Customize for your use case

Happy experimenting!
