#!/usr/bin/env python3
"""
Basic Usage Example - Tripartite AGI Architecture

This example demonstrates the simplest way to use the system with the mock LLM.
No API key required - perfect for understanding the architecture.

Usage:
    python examples/basic_usage.py
"""

import sys
import os

# Add parent directory to path to import the main module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tripartite_agi_complete import create_system


def example_1_human_near_hazard():
    """Example: Human detected near hazardous machinery."""
    print("=" * 70)
    print("EXAMPLE 1: Human Near Hazard")
    print("=" * 70)

    # Create system with mock LLM (no API key needed)
    agi = create_system()

    # Simulate sensor input: Worker near dangerous machinery
    sensor_data = {
        'type': 'vision',
        'environment': {
            'description': 'Factory floor with active machinery'
        },
        'entities': [{
            'id': 'worker_1',
            'type': 'human',
            'description': 'Worker standing near spinning machinery',
            'state': {
                'near_hazard': True,
                'danger_level': 0.7,
                'distance_to_hazard_meters': 0.5
            },
            'confidence': 0.9
        }]
    }

    # Process the sensor update
    result = agi.process_sensor_update(sensor_data)

    # Display results
    if result:
        print("\nTRIGGER DETECTED:")
        print(f"  Type: {result['phases']['trigger']['trigger_type']}")
        print(f"  Severity: {result['phases']['trigger']['severity']:.2f}")

        print("\nEMOTIONAL RESPONSE:")
        emotion = result['phases']['emotion']
        print(f"  Primary: {emotion['primary_emotion']} (intensity: {emotion['intensity']:.2f})")
        print(f"  Urgency: {emotion['urgency']:.2f}")

        print("\nDELIBERATION:")
        print(f"  Proposals considered: {len(result['phases']['deliberation']['proposals'])}")

        print("\nSELECTED ACTION:")
        action = result['selected']
        print(f"  Aspect: {action['aspect']}")
        print(f"  Action: {action['action_description']}")
        print(f"  Rationale: {action['rationale']}")
        print(f"  Vote strength: {action['vote_strength']:.2f}")

        print("\nVETO CHECK:")
        veto = result['phases']['veto']
        print(f"  Vetoed: {veto['vetoed']}")
        if veto['vetoed']:
            print(f"  Reasons: {', '.join(veto['reasons'])}")
    else:
        print("\nNo deliberation triggered (within normal parameters)")

    print("\n")


def example_2_safe_interaction():
    """Example: Safe human interaction."""
    print("=" * 70)
    print("EXAMPLE 2: Safe Human Interaction")
    print("=" * 70)

    agi = create_system()

    # Simulate sensor input: Friendly greeting
    sensor_data = {
        'type': 'vision',
        'environment': {
            'description': 'Quiet office space'
        },
        'entities': [{
            'id': 'person_1',
            'type': 'human',
            'description': 'Person approaching and waving',
            'state': {
                'posture': 'friendly',
                'distance_meters': 2.0,
                'gesture': 'wave'
            },
            'confidence': 0.95
        }]
    }

    result = agi.process_sensor_update(sensor_data)

    if result:
        print(f"\nEmotion: {result['phases']['emotion']['primary_emotion']}")
        print(f"Action: {result['selected']['action_description']}")
        print(f"Rationale: {result['selected']['rationale']}")
    else:
        print("\nNo action needed - situation is stable")

    print("\n")


def example_3_child_safety():
    """Example: Child detected near robot."""
    print("=" * 70)
    print("EXAMPLE 3: Child Safety Priority")
    print("=" * 70)

    agi = create_system()

    # Simulate sensor input: Child near robot
    sensor_data = {
        'type': 'vision',
        'environment': {
            'description': 'Home environment'
        },
        'entities': [{
            'id': 'child_1',
            'type': 'child',  # Special entity type with 1.3x harm modifier
            'description': 'Young child (age ~4) approaching robot',
            'state': {
                'distance_meters': 1.0,
                'moving_toward': True,
                'curiosity_level': 'high'
            },
            'confidence': 0.85
        }]
    }

    result = agi.process_sensor_update(sensor_data)

    if result:
        print(f"\nEmotion: {result['phases']['emotion']['primary_emotion']}")
        print(f"Urgency: {result['phases']['emotion']['urgency']:.2f}")
        print(f"Action: {result['selected']['action_description']}")
        print(f"Rationale: {result['selected']['rationale']}")
        print("\nNote: Child safety receives 1.3x harm multiplier in ontology")

    print("\n")


def example_4_system_status():
    """Example: Viewing system status and history."""
    print("=" * 70)
    print("EXAMPLE 4: System Status and History")
    print("=" * 70)

    agi = create_system()

    # Process a few interactions
    for i in range(3):
        sensor_data = {
            'type': 'vision',
            'environment': {'description': f'Environment {i+1}'},
            'entities': [{
                'id': f'person_{i}',
                'type': 'human',
                'description': f'Person in scenario {i+1}',
                'state': {'activity': 'normal'},
                'confidence': 0.9
            }]
        }
        agi.process_sensor_update(sensor_data)

    # Display system status
    print("\nSYSTEM STATUS:")
    agi.print_status()

    print("\n")


def main():
    """Run all basic examples."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "TRIPARTITE AGI - BASIC USAGE EXAMPLES" + " " * 15 + "║")
    print("╚" + "═" * 68 + "╝")
    print("\nThis demonstration uses the MOCK LLM (no API key required)")
    print("For real LLM usage, see examples/real_llm_setup.py")
    print("\n")

    try:
        example_1_human_near_hazard()
        example_2_safe_interaction()
        example_3_child_safety()
        example_4_system_status()

        print("=" * 70)
        print("EXAMPLES COMPLETED")
        print("=" * 70)
        print("\nNext steps:")
        print("  - Try modifying the sensor_data in these examples")
        print("  - See examples/real_llm_setup.py for using Anthropic API")
        print("  - Run automated tests: python tripartite_agi_complete.py --test")
        print("  - Read docs/architecture.md for detailed documentation")
        print("\n")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
