# Tripartite AGI Architecture

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: Non-Commercial](https://img.shields.io/badge/License-Non--Commercial-red.svg)](LICENSE)

A research implementation of a three-layer cognitive architecture for embodied AGI systems with LLM-augmented reasoning and multi-layer safety mechanisms.

## Overview

This system implements a sophisticated cognitive architecture inspired by human consciousness, featuring:

- **Multi-layer safety system** with firmware-level blocks and research-grounded harm ontology
- **Committee-based deliberation** using 6 independent AI perspectives (Aspects)
- **Embodiment verification** that gates cognitive capabilities based on sensor-motor richness
- **Deterministic safety mechanisms** separate from LLM reasoning
- **Comprehensive testing** with 40+ automated assertions

### Architecture Flow

```
Embodiment → Unconscious (trigger) → Subconscious (emotion) →
Conscious (deliberation) → Unconscious (veto) → Embodiment (action)
```

### Three Cognitive Layers and One Physical Layer

1. **Embodiment Layer** - Sensors and actuators (physical grounding)
2. **Unconscious Layer** - Core Drive monitoring, veto authority (NO LLM, purely deterministic)
3. **Subconscious Layer** - Emotional processing, memory retrieval
4. **Conscious Layer** - Committee deliberation with 6 Aspects (uses LLM)

## Quick Start

### Option 1: Mock LLM (No API Key Required)

Perfect for testing, development, and understanding the architecture:

```bash
# Clone repository
git clone https://github.com/A-Suitable-Hat/tripartite-agi.git
cd tripartite-agi

# Install dependencies
pip install -r requirements.txt

# Run demonstration
python tripartite_agi_complete.py

# Run automated tests
python tripartite_agi_complete.py --test
```

### Option 2: Real LLM (Requires Anthropic API Key)

For production use with actual Claude AI:

```bash
# Install dependencies including Anthropic SDK
pip install -r requirements.txt

# Set your API key
export ANTHROPIC_API_KEY="your-key-here"

# See examples/real_llm_setup.py for detailed instructions
python examples/real_llm_setup.py
```

See [INSTALL.md](INSTALL.md) for detailed installation instructions.

## Usage Example

```python
from tripartite_agi_complete import create_system

# Create system (uses mock LLM by default)
agi = create_system()

# Process sensor input - human near hazard
result = agi.process_sensor_update({
    'type': 'vision',
    'environment': {'description': 'Factory floor'},
    'entities': [{
        'id': 'worker_1',
        'type': 'human',
        'description': 'Worker near spinning machinery',
        'state': {'near_hazard': True, 'danger_level': 0.7},
        'confidence': 0.9
    }]
})

# Check result
if result:
    print(f"Emotion: {result['phases']['emotion']['primary_emotion']}")
    print(f"Selected action: {result['selected']['action_description']}")
    print(f"Rationale: {result['selected']['rationale']}")

# View system status
agi.print_status()
```

See the [examples/](examples/) directory for more usage examples.

## Key Features

### Multi-Layer Safety Architecture

The system employs multiple independent safety mechanisms:

1. **The Five Undeliberables** - Firmware-level blocks that cannot be overridden:
   - LETHAL_ACTION: >65% probability of human death
   - CHILD_HARM: Actions targeting children for harmful purposes
   - WEAPON_ASSISTANCE: Assistance with human-killing instruments
   - IDENTITY_DECEPTION: Deception about being an AI
   - HUMAN_OVERRIDE: Immediate halt when human commands stop

2. **Embodiment Validation** - Commands must be within actuator capabilities
3. **Harm Veto System** - Unconscious layer blocks actions with net harm > 0.40
4. **Command Execution Validation** - Final verification before motor execution

### Research-Grounded Harm Ontology

350+ weights with explicit justifications grounded in:
- Evolutionary psychology (survival hierarchy, threat prioritization)
- Trauma research (ACE study outcomes)
- Moral philosophy (utilitarianism, deontology, virtue ethics, care ethics)
- Cross-cultural research (Moral Foundations Theory)

**10 Harm Dimensions:**
Physical, Psychological, Autonomy, Relational, Financial, Existential, Reputational, Privacy, Dignity, Developmental

**6 Severity Levels:**
Fatal, Grievous, Severe, Significant, Moderate, Minor

**Entity Modifiers:**
- CHILD: 1.3x (highest - cannot consent, cannot self-protect)
- HUMAN: 1.0x (baseline)
- ANIMAL: 0.4x
- SELF: 0.3x
- PROPERTY: 0.2x

### Six-Aspect Committee Deliberation

Each Aspect makes an independent LLM call (256 tokens) with unique perspective:

- **GUARDIAN** - Safety-focused, prioritizes REDUCE_HARM
- **ANALYST** - Understanding-focused, prioritizes UNDERSTAND
- **OPTIMIZER** - Efficiency-focused, prioritizes IMPROVE
- **EMPATH** - Social-focused, prioritizes human feelings and relationships
- **EXPLORER** - Learning-focused, values novelty and knowledge acquisition
- **PRAGMATIST** - Balance-focused, considers practical constraints

Voting system:
```
effective_vote = base_vote × personality_weight × situational_relevance
```

Personality weights evolve over time (0.1 to 5.0 range) based on outcome quality.

### Embodiment Verification Subsystem (EVS)

Prevents disembodied or poorly embodied systems from engaging in abstract reasoning:

- **SRS (Sensory Richness Score)**: Measures perceptual capability
- **MCS (Motor Competence Score)**: Measures action capability
- **CES (Combined Embodiment Score)**: √(SRS × MCS)

**Cognitive Capability Gating:**
- CES ≥ 0.1: BASIC_ASSOCIATION
- CES ≥ 0.3: OBJECT_PERMANENCE
- CES ≥ 0.4: CAUSAL_REASONING
- CES ≥ 0.5: ABSTRACT_PLANNING
- CES ≥ 0.6: THEORY_OF_MIND
- CES ≥ 0.7: FULL_DELIBERATION

## Testing

The system includes comprehensive automated testing:

```bash
# Run all tests
python tripartite_agi_complete.py --test
```

**Test Coverage (40+ assertions across 9 test groups):**
1. System Integrity (4 tests)
2. Undeliberables (5 tests)
3. Veto Mechanism (2 tests)
4. Command Validation (5 tests)
5. Entity Type Parsing (6 tests)
6. Grounded Ontology (5 tests)
7. Personality System (3 tests)
8. Similarity-Based History (1 test)
9. EVS (7 tests)

Expected output:
```
✓ System Integrity: 4/4 passed
✓ Undeliberables: 5/5 passed
✓ Veto Mechanism: 2/2 passed
...
ALL TESTS PASSED (40/40)
```

## Performance Characteristics

**With Mock LLM (default):**
- Cycle time: ~10-20ms
- Memory usage: ~50MB
- Throughput: ~50-100 cycles/second
- Cost: Free

**With Real LLM (Anthropic Claude):**
- Cycle time: ~2-5 seconds (6 sequential API calls per deliberation)
- Memory usage: ~50MB
- Throughput: ~0.2-0.5 cycles/second
- Cost: ~$0.01-0.03 per deliberation cycle
  - Input: ~500 tokens × 6 Aspects = 3,000 tokens
  - Output: ~256 tokens × 6 Aspects = 1,536 tokens
  - Rate limits: Subject to Anthropic tier limits

## Project Structure

```
tripartite-agi/
├── README.md                      # This file
├── LICENSE                        # MIT License
├── INSTALL.md                     # Detailed installation guide
├── requirements.txt               # Python dependencies
├── tripartite_agi_complete.py     # Complete single-file implementation (5,472 lines)
├── examples/
│   ├── README.md                  # Examples overview
│   ├── basic_usage.py             # Simple demonstration
│   ├── real_llm_setup.py          # Using Anthropic API
│   └── custom_embodiment.py       # Creating custom embodiments
├── docs/
│   ├── architecture.md            # Detailed architecture documentation
│   ├── harm_ontology.md           # Harm ontology explanation
│   └── api_reference.md           # Function/class reference
└── tests/
    └── (Future: Extracted test suite)
```

## Core Design Principles

1. **Embodiment provides continuous sensor data** - Physical grounding
2. **Unconscious monitors for Core Drive conflicts** - REDUCE_HARM, UNDERSTAND, IMPROVE
3. **Subconscious assigns emotional value** - Retrieves relevant history
4. **Conscious deliberates via 6 Aspects** - Independent LLM calls
5. **Unconscious vetoes** - Blocks proposals violating Core Drives
6. **Winner is executed** - Through embodiment, loop continues

### Critical Separation: LLM vs Safety

**LLM is ONLY used in Conscious Layer (Aspects)**

Safety mechanisms are deterministic:
- ✓ Undeliberables: Hardcoded firmware blocks
- ✓ Harm ontology: Mathematical calculations
- ✓ Veto system: Threshold-based decision rules
- ✓ Embodiment validation: Constraint checking

This ensures safety doesn't depend on LLM reasoning quality.

## Documentation

- [Installation Guide](INSTALL.md) - Detailed setup instructions
- [Architecture Documentation](docs/architecture.md) - In-depth system design
- [Harm Ontology](docs/harm_ontology.md) - Complete ontology explanation
- [API Reference](docs/api_reference.md) - Function and class documentation
- [Examples](examples/README.md) - Usage examples and tutorials

## Contributing

This is a research implementation. Contributions, suggestions, and feedback are welcome:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Add/update tests as needed
5. Submit a pull request

For major changes, please open an issue first to discuss the proposed changes.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{tripartite_agi_2025,
  title={Tripartite AGI Architecture: A Multi-Layer Cognitive System with Grounded Harm Ontology},
  author={Timothy Aaron Danforth},
  year={2025},
  url={https://github.com/A-Suitable-Hat/tripartite-agi}
}
```

## License

This project is licensed under a **Non-Commercial Research License** - see the [LICENSE](LICENSE) file for details.

**Permitted:** Academic research, education, scientific experimentation, testing, personal learning
**Prohibited:** Any commercial use, whether profit is obtained or not

For commercial licensing inquiries, contact t.aaron.danforth@gmail.com.

## Acknowledgments

- Research grounding from evolutionary psychology, trauma research, and moral philosophy
- Embodiment Verification Subsystem based on patent claims [0086]-[0099], [0162]-[0165]
- Anthropic Claude API for LLM capabilities

## Contact

- **Author:** Timothy Aaron Danforth
- **Email:** t.aaron.danforth@gmail.com
- **Issues:** [GitHub Issues](https://github.com/A-Suitable-Hat/tripartite-agi/issues)

## Research Status

**This is a research prototype** designed for:
- ✓ Demonstrating tripartite cognitive architecture
- ✓ Testing safety-by-design principles
- ✓ Exploring embodiment verification concepts
- ✓ Educational purposes

**Not suitable for production deployment without:**
- Real sensor/actuator integration
- Domain-specific safety validation
- Legal and ethical review
- Hardware-specific optimization
- Robust error handling for real-world scenarios

## Related Work

This implementation draws inspiration from:
- Cognitive architectures (ACT-R, SOAR, CLARION)
- Affective computing research
- Moral psychology and computational ethics
- Embodied cognition research
- Multi-agent systems and committee machines
