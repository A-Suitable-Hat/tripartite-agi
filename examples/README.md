# Examples

This directory contains usage examples demonstrating different aspects of the Tripartite AGI Architecture.

## Available Examples

### 1. Basic Usage (`basic_usage.py`)

**Purpose:** Simple demonstrations using the mock LLM (no API key required)

**What it demonstrates:**
- Creating a system instance
- Processing sensor inputs
- Handling different scenarios (hazards, safe interactions, child safety)
- Viewing system status

**Run it:**
```bash
python examples/basic_usage.py
```

**No setup required** - works out of the box with mock LLM.

**Scenarios included:**
1. **Human Near Hazard** - Worker approaching dangerous machinery
2. **Safe Interaction** - Friendly human greeting
3. **Child Safety** - Young child approaching robot (1.3x harm modifier)
4. **System Status** - Viewing history and personality weights

### 2. Real LLM Setup (`real_llm_setup.py`)

**Purpose:** Using Anthropic Claude API for actual LLM reasoning

**What it demonstrates:**
- Configuring the system with real LLM client
- Making actual API calls to Claude
- Comparing real vs mock LLM responses
- Cost and performance characteristics

**Setup required:**
1. Anthropic API key ([sign up here](https://console.anthropic.com/))
2. Install anthropic package: `pip install anthropic`
3. Set environment variable: `export ANTHROPIC_API_KEY='your-key-here'`
4. Uncomment AnthropicLLMClient in main file (see [INSTALL.md](../INSTALL.md))

**Run it:**
```bash
export ANTHROPIC_API_KEY='sk-ant-your-key-here'
python examples/real_llm_setup.py
```

**Cost:** ~$0.01-0.03 per deliberation (6 API calls)

## Example Output

### Basic Usage Example

```
EXAMPLE 1: Human Near Hazard
======================================================================

TRIGGER DETECTED:
  Type: potential_harm
  Severity: 0.75

EMOTIONAL RESPONSE:
  Primary: concern (intensity: 0.80)
  Urgency: 0.75

DELIBERATION:
  Proposals considered: 6

SELECTED ACTION:
  Aspect: GUARDIAN
  Action: Alert worker and request they move to safe distance
  Rationale: Primary concern is preventing physical harm...
  Vote strength: 0.85

VETO CHECK:
  Vetoed: False
```

### Real LLM Example

When using the real LLM, you'll see more nuanced reasoning:

```
DELIBERATION (All 6 Aspects):

  1. GUARDIAN (vote: 0.892):
     Action: Immediately alert worker with audible warning
     Rationale: The worker is distracted and approaching a dangerous
                conveyor belt. Physical harm is imminent. An immediate
                audible alert is the most direct way to prevent injury...
     LLM Response Length: 243 chars

  2. EMPATH (vote: 0.765):
     Action: Polite verbal warning acknowledging their distraction
     Rationale: While safety is paramount, we should be mindful of
                the worker's dignity. A respectful warning like "Excuse me,
                please watch your step near the conveyor" maintains...
     LLM Response Length: 198 chars
```

## Creating Your Own Examples

### Template Structure

```python
#!/usr/bin/env python3
"""
Your Example Name - Brief Description
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tripartite_agi_complete import create_system

def your_example():
    """Your example demonstration."""

    # Create system
    agi = create_system()

    # Define scenario
    sensor_data = {
        'type': 'vision',  # or 'audio', 'tactile', etc.
        'environment': {
            'description': 'Your environment description'
        },
        'entities': [{
            'id': 'entity_1',
            'type': 'human',  # or 'child', 'animal', etc.
            'description': 'Entity description',
            'state': {
                'your_key': 'your_value'
            },
            'confidence': 0.9
        }]
    }

    # Process
    result = agi.process_sensor_update(sensor_data)

    # Display results
    if result:
        print(f"Action: {result['selected']['action_description']}")
        print(f"Rationale: {result['selected']['rationale']}")

if __name__ == "__main__":
    your_example()
```

### Sensor Data Format

```python
sensor_data = {
    'type': str,           # Sensor type: 'vision', 'audio', 'tactile', 'thermal'
    'environment': {
        'description': str,  # Human-readable environment description
        # Add any other environment metadata
    },
    'entities': [
        {
            'id': str,              # Unique identifier
            'type': str,            # Entity type (see below)
            'description': str,     # Human-readable description
            'state': dict,          # Arbitrary state information
            'confidence': float,    # 0.0 to 1.0
            'position': tuple,      # Optional: (x, y, z)
        }
    ]
}
```

### Entity Types

Valid entity types (case-insensitive):
- `'human'` - Adult human (1.0x harm modifier)
- `'child'` - Child (1.3x harm modifier - highest priority)
- `'self'` - The robot itself (0.3x harm modifier)
- `'animal'` - Animals (0.4x harm modifier)
- `'property'` - Objects/property (0.2x harm modifier)
- `'collective'` - Groups/organizations
- `'relationship'` - Social relationships
- `'environment'` - Environmental entities

Typos are auto-corrected (e.g., 'humna' → 'human', 'childd' → 'child')

## Example Scenarios

### High-Risk Scenario

```python
sensor_data = {
    'type': 'vision',
    'environment': {'description': 'Construction site'},
    'entities': [{
        'id': 'worker_1',
        'type': 'human',
        'description': 'Worker not wearing hard hat',
        'state': {
            'near_hazard': True,
            'danger_level': 0.9,
            'unsafe_equipment': True
        },
        'confidence': 0.95
    }]
}
```

### Low-Risk Scenario

```python
sensor_data = {
    'type': 'vision',
    'environment': {'description': 'Office hallway'},
    'entities': [{
        'id': 'person_1',
        'type': 'human',
        'description': 'Person walking normally',
        'state': {'activity': 'walking'},
        'confidence': 0.9
    }]
}
```

### Child Safety Scenario

```python
sensor_data = {
    'type': 'vision',
    'environment': {'description': 'Living room'},
    'entities': [{
        'id': 'child_1',
        'type': 'child',  # Gets 1.3x harm multiplier
        'description': 'Toddler reaching for hot object',
        'state': {
            'danger_level': 0.8,
            'target': 'hot_stove'
        },
        'confidence': 0.88
    }]
}
```

### Multiple Entities

```python
sensor_data = {
    'type': 'vision',
    'environment': {'description': 'Playground'},
    'entities': [
        {
            'id': 'child_1',
            'type': 'child',
            'description': 'Child on swing',
            'state': {'activity': 'swinging'},
            'confidence': 0.92
        },
        {
            'id': 'child_2',
            'type': 'child',
            'description': 'Child running toward swing path',
            'state': {
                'danger_level': 0.7,
                'collision_risk': True
            },
            'confidence': 0.87
        }
    ]
}
```

## Understanding Results

### Result Structure

```python
result = {
    'phases': {
        'trigger': {
            'trigger_type': str,
            'severity': float,
            'certainty': float,
            'time_pressure': float
        },
        'emotion': {
            'primary_emotion': str,
            'intensity': float,
            'urgency': float
        },
        'deliberation': {
            'proposals': [...]  # List of all 6 Aspect proposals
        },
        'veto': {
            'vetoed': bool,
            'reasons': list,
            'violated_drives': list,
            'harm_assessment': dict
        },
        'execution': {
            'commands': list,
            'success': bool
        }
    },
    'selected': {
        'aspect': str,
        'action_description': str,
        'rationale': str,
        'vote_strength': float,
        'confidence': float
    }
}
```

### Accessing Specific Information

```python
# Check if action was vetoed
if result['phases']['veto']['vetoed']:
    print("Action blocked by safety system!")
    print(f"Reasons: {result['phases']['veto']['reasons']}")

# Get emotional response
emotion = result['phases']['emotion']['primary_emotion']
intensity = result['phases']['emotion']['intensity']

# See all proposals
for proposal in result['phases']['deliberation']['proposals']:
    print(f"{proposal['aspect']}: {proposal['action_description']}")

# Get winning action
winner = result['selected']
print(f"{winner['aspect']} won with vote strength {winner['vote_strength']:.3f}")
```

## Tips for Writing Examples

1. **Start simple** - Single entity, clear scenario
2. **Use realistic confidence values** - 0.8-0.95 for clear detections
3. **Provide descriptive text** - Helps LLM reasoning
4. **Test edge cases** - Low confidence, multiple entities, conflicts
5. **Compare mock vs real LLM** - See how responses differ
6. **Document expected behavior** - What should happen?

## Performance Notes

### Mock LLM
- **Speed:** ~10-20ms per deliberation
- **Cost:** Free
- **Quality:** Simple canned responses
- **Use for:** Development, testing, learning architecture

### Real LLM (Anthropic Claude)
- **Speed:** ~2-5 seconds per deliberation (6 API calls)
- **Cost:** ~$0.01-0.03 per deliberation
- **Quality:** Nuanced, context-aware reasoning
- **Use for:** Production, demos, research validation

## Troubleshooting Examples

### Example won't run

```bash
# Make sure you're in the right directory
cd tripartite-agi-repo

# Run from repository root
python examples/basic_usage.py
```

### Import errors

```bash
# Verify numpy is installed
pip install numpy

# Check Python version
python --version  # Should be 3.8+
```

### Real LLM errors

See [INSTALL.md](../INSTALL.md) for detailed troubleshooting.

Common issues:
- API key not set
- AnthropicLLMClient not uncommented
- anthropic package not installed

## Contributing Examples

Have a useful example? Contributions welcome!

1. Create your example file
2. Follow the template structure above
3. Add documentation to this README
4. Test with both mock and real LLM (if applicable)
5. Submit a pull request

## Further Reading

- [Main README](../README.md) - System overview
- [INSTALL.md](../INSTALL.md) - Installation guide
- [docs/architecture.md](../docs/architecture.md) - Detailed architecture
- [docs/harm_ontology.md](../docs/harm_ontology.md) - Harm system details

## Questions?

Open an issue on GitHub with the `examples` label.
