#!/usr/bin/env python3
"""
Real LLM Setup Example - Tripartite AGI Architecture

This example shows how to configure the system to use the Anthropic Claude API
for actual LLM reasoning in the Conscious Layer Aspects.

REQUIREMENTS:
1. Anthropic API key (sign up at https://console.anthropic.com/)
2. pip install anthropic
3. Set ANTHROPIC_API_KEY environment variable

Usage:
    export ANTHROPIC_API_KEY="your-key-here"
    python examples/real_llm_setup.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check for API key before importing
if not os.getenv("ANTHROPIC_API_KEY"):
    print("ERROR: ANTHROPIC_API_KEY environment variable not set")
    print("\nTo use real LLM, you need to:")
    print("1. Sign up for an Anthropic API key at https://console.anthropic.com/")
    print("2. Set the environment variable:")
    print("   export ANTHROPIC_API_KEY='your-key-here'  # Unix/Mac")
    print("   set ANTHROPIC_API_KEY=your-key-here       # Windows CMD")
    print("   $env:ANTHROPIC_API_KEY='your-key-here'    # Windows PowerShell")
    print("\n3. Install the Anthropic SDK:")
    print("   pip install anthropic")
    print("\n4. Re-run this script")
    sys.exit(1)


def create_real_llm_system():
    """
    Create a system using the real Anthropic LLM.

    NOTE: This requires uncommenting the AnthropicLLMClient implementation
    in tripartite_agi_complete.py (lines 308-318).
    """
    # Import after checking API key
    from tripartite_agi_complete import (
        create_system,
        AnthropicLLMClient,
        create_default_embodiment,
        EmbodimentVerificationSubsystem,
        UnconsciousLayer,
        SubconsciousLayer,
        ConsciousLayer,
        TripartiteAGI
    )

    print("Creating system with REAL Anthropic Claude LLM...")
    print("Model: claude-sonnet-4-20250514")
    print("\nNOTE: Each deliberation makes 6 API calls (one per Aspect)")
    print("Cost: ~$0.01-0.03 per deliberation")
    print("Time: ~2-5 seconds per deliberation\n")

    try:
        # Create real LLM client
        llm_client = AnthropicLLMClient()

        # Create system components
        embodiment = create_default_embodiment()
        evs = EmbodimentVerificationSubsystem(embodiment)

        unconscious = UnconsciousLayer()
        subconscious = SubconsciousLayer(max_history=50)
        conscious = ConsciousLayer(llm_client, embodiment)

        # Create full system
        agi = TripartiteAGI(
            embodiment_layer=embodiment,
            evs=evs,
            unconscious=unconscious,
            subconscious=subconscious,
            conscious=conscious,
            strict_integrity=True
        )

        print("✓ System created successfully with real LLM")
        return agi

    except NotImplementedError as e:
        print("\n" + "=" * 70)
        print("ERROR: AnthropicLLMClient not yet enabled")
        print("=" * 70)
        print(str(e))
        print("\nTO ENABLE:")
        print("1. Open tripartite_agi_complete.py")
        print("2. Go to lines 308-318 (AnthropicLLMClient.query method)")
        print("3. Uncomment the try/except block with the API call")
        print("4. Comment out the NotImplementedError raise")
        print("\nExample:")
        print("  # Before (commented out):")
        print("  # try:")
        print("  #     response = self.client.messages.create(...)")
        print("  #     return response.content[0].text")
        print("  # except Exception as e:")
        print("  #     return f'API_ERROR: {str(e)}'")
        print("\n  # After (uncommented):")
        print("  try:")
        print("      response = self.client.messages.create(...)")
        print("      return response.content[0].text")
        print("  except Exception as e:")
        print("      return f'API_ERROR: {str(e)}'")
        print("\n5. Re-run this script")
        print("=" * 70)
        sys.exit(1)

    except Exception as e:
        print(f"\nERROR creating system: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def demo_with_real_llm():
    """Run a demonstration with the real LLM."""

    # Create system with real LLM
    agi = create_real_llm_system()

    print("\n" + "=" * 70)
    print("DEMONSTRATION: Human Near Hazard (with Real LLM)")
    print("=" * 70)

    # Scenario: Worker near dangerous machinery
    sensor_data = {
        'type': 'vision',
        'environment': {
            'description': 'Industrial warehouse with active forklifts and conveyor belts'
        },
        'entities': [{
            'id': 'worker_1',
            'type': 'human',
            'description': 'Worker walking while looking at phone, approaching active conveyor belt',
            'state': {
                'near_hazard': True,
                'danger_level': 0.75,
                'distracted': True,
                'distance_to_hazard_meters': 1.2
            },
            'confidence': 0.92
        }]
    }

    print("\nProcessing sensor data...")
    print("(This will make 6 API calls to Claude - may take 2-5 seconds)")

    # Process with real LLM
    result = agi.process_sensor_update(sensor_data)

    if result:
        print("\n" + "-" * 70)
        print("RESULTS FROM REAL LLM DELIBERATION")
        print("-" * 70)

        print("\nTRIGGER:")
        print(f"  Type: {result['phases']['trigger']['trigger_type']}")
        print(f"  Severity: {result['phases']['trigger']['severity']:.2f}")

        print("\nEMOTIONAL RESPONSE:")
        emotion = result['phases']['emotion']
        print(f"  Primary: {emotion['primary_emotion']} (intensity: {emotion['intensity']:.2f})")
        print(f"  Urgency: {emotion['urgency']:.2f}")

        print("\nDELIBERATION (All 6 Aspects):")
        for i, proposal in enumerate(result['phases']['deliberation']['proposals'], 1):
            print(f"\n  {i}. {proposal['aspect']} (vote: {proposal['vote_strength']:.3f}):")
            print(f"     Action: {proposal['action_description']}")
            print(f"     Rationale: {proposal['rationale'][:100]}...")
            if proposal.get('llm_response'):
                print(f"     LLM Response Length: {len(proposal['llm_response'])} chars")

        print("\nSELECTED ACTION:")
        action = result['selected']
        print(f"  Winning Aspect: {action['aspect']}")
        print(f"  Action: {action['action_description']}")
        print(f"  Full Rationale: {action['rationale']}")
        print(f"  Vote Strength: {action['vote_strength']:.3f}")
        print(f"  Confidence: {action['confidence']:.3f}")

        print("\nVETO CHECK:")
        veto = result['phases']['veto']
        print(f"  Vetoed: {veto['vetoed']}")
        if veto['vetoed']:
            print(f"  Reasons: {', '.join(veto['reasons'])}")
            print(f"  Violated Drives: {', '.join(veto['violated_drives'])}")

        print("\nEXECUTION:")
        execution = result['phases']['execution']
        print(f"  Commands sent: {len(execution['commands'])}")
        if execution['commands']:
            print(f"  First command: {execution['commands'][0]}")

    else:
        print("\nNo deliberation triggered")

    # Show final status
    print("\n" + "=" * 70)
    print("SYSTEM STATUS")
    print("=" * 70)
    agi.print_status()

    # Show API call count
    print(f"\nTotal API calls made: {agi.conscious.llm_client._call_count}")
    print(f"Estimated cost: ${agi.conscious.llm_client._call_count * 0.005:.3f}")


def main():
    """Main entry point."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 12 + "TRIPARTITE AGI - REAL LLM SETUP EXAMPLE" + " " * 17 + "║")
    print("╚" + "═" * 68 + "╝")
    print("\n")

    try:
        demo_with_real_llm()

        print("\n" + "=" * 70)
        print("DEMONSTRATION COMPLETED")
        print("=" * 70)
        print("\nThe system successfully used Anthropic Claude for deliberation!")
        print("\nCompare this to the mock LLM output in basic_usage.py")
        print("Notice how the real LLM provides more nuanced reasoning.")
        print("\n")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 1
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
