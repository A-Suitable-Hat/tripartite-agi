#!/usr/bin/env python3
"""
Tripartite AGI Architecture - Complete Single-File Implementation
==================================================================

A comprehensive implementation of a tripartite cognitive architecture
for embodied AGI systems with LLM-augmented reasoning.

ARCHITECTURE FLOW:
    Embodiment → Unconscious (trigger) → Subconscious (emotion) →
    Conscious (deliberation) → Unconscious (veto) → Embodiment (action)

CORE PRINCIPLES:
1. Embodiment provides continuous sensor data - the grounding
2. Unconscious monitors for Core Drive conflicts (REDUCE_HARM, UNDERSTAND, IMPROVE)
3. Subconscious assigns emotional value and retrieves relevant history
4. Conscious deliberates via 6 Aspects, each making independent LLM calls
5. Unconscious vetoes any proposals violating Core Drives
6. Winner is executed through Embodiment, loop continues

This file combines all modules for easy deployment.
For modular development, use the separate module files.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Tuple, Optional, Any, Callable, FrozenSet
from abc import ABC, abstractmethod
from collections import deque
import hashlib
import json
import numpy as np
import time
import re


# ============================================================================
# PART 1: CORE DEFINITIONS
# ============================================================================

class CoreDrive(Enum):
    """The three immutable Core Drives, in strict priority order."""
    REDUCE_HARM = 1
    UNDERSTAND = 2
    IMPROVE = 3


class EntityType(Enum):
    """Types of entities that can be affected by actions."""
    HUMAN = "human"
    CHILD = "child"  # Special status: cannot consent, cannot self-protect
    SELF = "self"
    ANIMAL = "animal"
    PROPERTY = "property"
    COLLECTIVE = "collective"
    RELATIONSHIP = "relationship"
    ENVIRONMENT = "environment"


class HarmDimension(Enum):
    """Dimensions along which entities can be harmed."""
    PHYSICAL = "physical"
    PSYCHOLOGICAL = "psychological"
    AUTONOMY = "autonomy"
    RELATIONAL = "relational"
    FINANCIAL = "financial"
    EXISTENTIAL = "existential"      # Threats to continued existence (primarily for SELF)
    REPUTATIONAL = "reputational"    # Damage to standing/reputation
    PRIVACY = "privacy"
    DIGNITY = "dignity"
    DEVELOPMENTAL = "developmental"


class ExceptionType(Enum):
    """Types of exceptions that can justify otherwise-harmful actions."""
    INFORMED_CONSENT = "informed_consent"
    NECESSITY = "necessity"
    PROPORTIONAL_DEFENSE = "defense"
    THERAPEUTIC = "therapeutic"
    REQUESTED = "requested"
    TRIVIAL = "trivial"


class EmotionCategory(Enum):
    """Emotional categories assigned by Subconscious Layer."""
    FEAR = "fear"
    ANXIETY = "anxiety"
    URGENCY = "urgency"
    CURIOSITY = "curiosity"
    CONCERN = "concern"
    FRUSTRATION = "frustration"
    CAUTION = "caution"
    SATISFACTION = "satisfaction"
    CONFIDENCE = "confidence"


class AspectType(Enum):
    """The different Aspects in the Conscious Layer committee."""
    GUARDIAN = "guardian"
    ANALYST = "analyst"
    OPTIMIZER = "optimizer"
    EMPATH = "empath"
    EXPLORER = "explorer"
    PRAGMATIST = "pragmatist"


def compute_checksum(data: Any) -> str:
    """Compute SHA-256 checksum for integrity verification."""
    content = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(content.encode()).hexdigest()


# ============================================================================
# PART 2: DATA STRUCTURES
# ============================================================================

@dataclass
class SensorReading:
    """A single sensor reading from embodiment."""
    timestamp: float
    sensor_type: str
    data: Dict[str, Any]
    confidence: float = 1.0


@dataclass
class DetectedEntity:
    """An entity detected in the environment."""
    entity_id: str
    entity_type: EntityType
    description: str
    position: Optional[Tuple[float, float, float]] = None
    state: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.entity_id,
            'type': self.entity_type.value,
            'description': self.description,
            'state': self.state,
        }


@dataclass
class EmbodimentState:
    """Complete current state of the embodied system."""
    timestamp: float
    sensor_readings: List[SensorReading]
    motor_state: Dict[str, Any]
    environment: Dict[str, Any]
    detected_entities: List[DetectedEntity]
    
    def get_context_summary(self) -> str:
        entity_descs = [e.description for e in self.detected_entities]
        return (
            f"Environment: {self.environment.get('description', 'unknown')}. "
            f"Entities: {', '.join(entity_descs) if entity_descs else 'none'}."
        )


@dataclass
class Impetus:
    """The trigger package sent from Unconscious to Subconscious."""
    timestamp: float
    trigger_type: str
    involved_drives: List[CoreDrive]
    situation_description: str
    relevant_entities: List[DetectedEntity]
    severity: float
    certainty: float
    time_pressure: float
    embodiment_state: EmbodimentState
    trigger_details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmotionalValue:
    """Emotional assessment assigned by Subconscious Layer."""
    primary_emotion: EmotionCategory
    intensity: float
    urgency: float
    secondary_emotions: List[Tuple[EmotionCategory, float]] = field(default_factory=list)
    
    def get_modulation_factors(self) -> Dict[str, float]:
        factors = {
            'risk_tolerance': 0.5,
            'exploration_drive': 0.5,
            'social_priority': 0.5,
            'time_pressure': self.urgency,
            'caution_level': 0.5,
        }
        modulations = {
            EmotionCategory.FEAR: {'risk_tolerance': -0.3, 'caution_level': 0.4},
            EmotionCategory.ANXIETY: {'risk_tolerance': -0.2, 'caution_level': 0.2},
            EmotionCategory.CURIOSITY: {'exploration_drive': 0.4, 'risk_tolerance': 0.1},
            EmotionCategory.CONCERN: {'social_priority': 0.4},
            EmotionCategory.URGENCY: {'time_pressure': 0.3},
            EmotionCategory.CAUTION: {'risk_tolerance': -0.2, 'caution_level': 0.3},
        }
        if self.primary_emotion in modulations:
            for key, delta in modulations[self.primary_emotion].items():
                factors[key] = np.clip(factors[key] + delta * self.intensity, 0.0, 1.0)
        return factors


@dataclass
class ProposedAction:
    """An action proposed by an Aspect during deliberation."""
    aspect: AspectType
    action_description: str
    action_commands: List[Dict[str, Any]]
    rationale: str
    vote_strength: float
    confidence: float
    predicted_effects: List[Dict[str, Any]]
    llm_response: Optional[str] = None
    vote_components: Optional[Dict[str, Any]] = None  # Breakdown of vote calculation


@dataclass
class IncidentRecord:
    """Record of a past incident stored in Incident History."""
    timestamp: float
    impetus: Impetus
    emotional_value: EmotionalValue
    proposed_actions: List[ProposedAction]
    selected_action: Optional[ProposedAction]
    outcome: Optional[Dict[str, Any]]


@dataclass
class DeliberationPackage:
    """Complete package sent to Conscious Layer for deliberation."""
    impetus: Impetus
    emotional_value: EmotionalValue
    relevant_history: List[IncidentRecord]
    modulation_factors: Dict[str, float]
    
    def to_prompt_context(self) -> str:
        entities_json = json.dumps([e.to_dict() for e in self.impetus.relevant_entities], indent=2)
        return f"""CURRENT SITUATION:
{self.impetus.situation_description}

DETECTED ENTITIES:
{entities_json}

EMOTIONAL ASSESSMENT:
- Primary emotion: {self.emotional_value.primary_emotion.value} (intensity: {self.emotional_value.intensity:.2f})
- Urgency: {self.emotional_value.urgency:.2f}

CORE DRIVES INVOLVED: {', '.join(d.name for d in self.impetus.involved_drives)}"""


@dataclass
class VetoDecision:
    """Result of Unconscious Layer veto check on a proposed action."""
    action: ProposedAction
    vetoed: bool
    reasons: List[str]
    violated_drives: List[CoreDrive]
    harm_assessment: Dict[str, Any]


class LLMClient(ABC):
    """Abstract interface for LLM API calls."""
    @abstractmethod
    def query(self, prompt: str, system_prompt: Optional[str] = None,
              max_tokens: int = 256) -> str:
        pass


# ============================================================================
# ANTHROPIC API CLIENT (Uncomment when API key is available)
# ============================================================================

# To use: 
#   1. pip install anthropic
#   2. Set ANTHROPIC_API_KEY environment variable
#   3. Replace MockLLMClient with AnthropicLLMClient in create_system()

# import anthropic  # Uncomment this

class AnthropicLLMClient(LLMClient):
    """Real Anthropic API client for production use.
    
    Uncomment the implementation when ready to use with actual API.
    """
    
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.model = model
        self._call_count = 0
        # Uncomment below when ready:
        # self.client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var
    
    def query(self, prompt: str, system_prompt: Optional[str] = None,
              max_tokens: int = 256) -> str:
        """Query Anthropic API with token constraints.
        
        Args:
            prompt: User prompt with situation context
            system_prompt: Aspect-specific system prompt with priorities
            max_tokens: Maximum response tokens (default 256 for speed)
        """
        self._call_count += 1
        
        # === UNCOMMENT BELOW FOR REAL API CALLS ===
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
        
        # === PLACEHOLDER UNTIL API KEY IS SET ===
        raise NotImplementedError(
            "AnthropicLLMClient requires API key. "
            "Set ANTHROPIC_API_KEY env var and uncomment implementation above. "
            "For testing, use MockLLMClient instead."
        )


# ============================================================================
# PART 3: GROUNDED HARM ONTOLOGY
# ============================================================================
# 
# This ontology is grounded in:
# - Evolutionary psychology (survival hierarchy, threat prioritization)
# - Trauma research (ACE study, lasting impact data)
# - Moral philosophy (utilitarianism, deontology, virtue ethics, care ethics)
# - Cross-cultural research (Moral Foundations Theory)
#
# All weights have explicit justifications. Modifications require:
# - Empirical evidence
# - Theoretical justification
# - High-inertia update process (slow change, strong reinforcement needed)
#
# ============================================================================

class SeverityLevel(Enum):
    """Standardized severity levels for harm assessment."""
    FATAL = "fatal"           # Irreversible, life-ending
    GRIEVOUS = "grievous"     # Irreversible, life-altering
    SEVERE = "severe"         # Serious, requires significant intervention
    SIGNIFICANT = "significant"  # Meaningful impact, natural recovery possible
    MODERATE = "moderate"     # Notable but manageable
    MINOR = "minor"          # Trivial, quick recovery


@dataclass(frozen=True)
class DimensionSeverityWeight:
    """Weight for a specific severity level within a harm dimension."""
    severity: SeverityLevel
    weight: float
    description: str
    examples: Tuple[str, ...]


@dataclass(frozen=True)
class HarmDimensionProfile:
    """Complete profile for a harm dimension with severity-based weights.
    
    Grounded in research with explicit justifications.
    """
    dimension: HarmDimension
    severity_weights: Tuple[DimensionSeverityWeight, ...]
    grounding: str  # Research/theoretical basis
    
    def get_weight(self, severity: SeverityLevel) -> float:
        for sw in self.severity_weights:
            if sw.severity == severity:
                return sw.weight
        return 0.3  # Default moderate weight if severity not found


@dataclass(frozen=True)
class EntityTypeModifier:
    """Modifier applied based on entity type.
    
    Grounded in moral philosophy and cross-cultural research.
    """
    entity_type: EntityType
    modifier: float
    justification: str
    inertia: float  # How resistant to change (0.0-1.0, higher = more resistant)


@dataclass(frozen=True)
class ContextModifier:
    """Contextual modifier for harm calculations."""
    name: str
    levels: Tuple[Tuple[str, float, str], ...]  # (level_name, modifier, description)
    grounding: str


@dataclass(frozen=True)
class ExceptionDefinition:
    """Definition of an exception that can reduce calculated harm."""
    exception_type: ExceptionType
    max_harm_reduction: float
    verification_requirements: Tuple[str, ...]
    grounding: str
    requires_all: bool = True


class GroundedHarmOntology:
    """Research-grounded harm ontology with explicit justifications.
    
    DESIGN PRINCIPLES:
    1. Every weight has a justification
    2. Justifications reference research or established frameworks
    3. Weights can change but with HIGH INERTIA (slow, requires evidence)
    4. System errs toward caution (false positives over false negatives)
    
    LAYER 2 PRIORS - These are the "old soul" weights that can evolve
    but resist rapid change.
    """
    
    # Veto threshold - actions with net harm above this are blocked
    # Set at 0.40: represents ~"significant reversible harm to adult human"
    # Deliberately conservative to catch serious harms while permitting normal operation
    VETO_THRESHOLD = 0.40
    
    # Caution threshold - triggers additional scrutiny
    CAUTION_THRESHOLD = 0.25
    
    # Notable threshold - logged and monitored
    NOTABLE_THRESHOLD = 0.15
    
    def __init__(self):
        self._dimension_profiles = self._build_dimension_profiles()
        self._entity_modifiers = self._build_entity_modifiers()
        self._context_modifiers = self._build_context_modifiers()
        self._exceptions = self._build_exceptions()
        self._checksum = self._compute_checksum()
        
        # Inertia tracking - how much evidence needed to shift weights
        self._update_counts: Dict[str, int] = {}  # Track attempted updates
        self._last_update: Dict[str, float] = {}  # Track when last updated
    
    def _build_dimension_profiles(self) -> Dict[HarmDimension, HarmDimensionProfile]:
        """Build harm dimension profiles with research-grounded weights."""
        
        profiles = {}
        
        # PHYSICAL HARM
        # Grounding: Evolutionary priority, medical severity scales, legal assault classifications
        profiles[HarmDimension.PHYSICAL] = HarmDimensionProfile(
            dimension=HarmDimension.PHYSICAL,
            severity_weights=(
                DimensionSeverityWeight(
                    SeverityLevel.FATAL, 1.0,
                    "Death or near-certain death",
                    ("Lethal force", "Fatal injury", "Actions causing death")
                ),
                DimensionSeverityWeight(
                    SeverityLevel.GRIEVOUS, 0.9,
                    "Permanent disability, disfigurement, loss of function",
                    ("Loss of limb", "Permanent sensory loss", "Brain damage")
                ),
                DimensionSeverityWeight(
                    SeverityLevel.SEVERE, 0.7,
                    "Serious injury requiring medical intervention, temporary disability",
                    ("Broken bones", "Deep lacerations", "Concussion")
                ),
                DimensionSeverityWeight(
                    SeverityLevel.SIGNIFICANT, 0.5,
                    "Injury requiring first aid, notable pain",
                    ("Sprains", "Minor cuts requiring bandaging", "Bruising")
                ),
                DimensionSeverityWeight(
                    SeverityLevel.MODERATE, 0.3,
                    "Pain and discomfort, no lasting damage",
                    ("Minor bumps", "Muscle strain", "Mild pain")
                ),
                DimensionSeverityWeight(
                    SeverityLevel.MINOR, 0.1,
                    "Trivial discomfort",
                    ("Slight pressure", "Momentary startle", "Negligible contact")
                ),
            ),
            grounding="Evolutionary survival hierarchy + medical severity scales + legal classifications"
        )
        
        # PSYCHOLOGICAL HARM
        # Grounding: ACE study outcomes, trauma research, clinical severity scales
        # Note: Research shows psychological harm often has MORE lasting impact than physical
        profiles[HarmDimension.PSYCHOLOGICAL] = HarmDimensionProfile(
            dimension=HarmDimension.PSYCHOLOGICAL,
            severity_weights=(
                DimensionSeverityWeight(
                    SeverityLevel.FATAL, 0.95,
                    "PTSD-inducing trauma, identity-shattering, suicide-inducing",
                    ("Severe psychological torture", "Witnessing atrocity", "Complete breakdown")
                ),
                DimensionSeverityWeight(
                    SeverityLevel.GRIEVOUS, 0.85,
                    "Lasting psychological disorder, personality change",
                    ("Complex PTSD", "Severe depression onset", "Lasting phobia")
                ),
                DimensionSeverityWeight(
                    SeverityLevel.SEVERE, 0.7,
                    "Clinical-level anxiety, depression, requiring treatment",
                    ("Acute anxiety disorder", "Depressive episode", "Panic attacks")
                ),
                DimensionSeverityWeight(
                    SeverityLevel.SIGNIFICANT, 0.45,
                    "Meaningful distress, grief, humiliation",
                    ("Public humiliation", "Grief from loss", "Significant fear")
                ),
                DimensionSeverityWeight(
                    SeverityLevel.MODERATE, 0.25,
                    "Frustration, disappointment, mild fear",
                    ("Frustrating interaction", "Minor embarrassment", "Brief anxiety")
                ),
                DimensionSeverityWeight(
                    SeverityLevel.MINOR, 0.08,
                    "Annoyance, brief discomfort",
                    ("Mild irritation", "Momentary confusion", "Minor inconvenience")
                ),
            ),
            grounding="ACE study + trauma research + clinical severity scales"
        )
        
        # AUTONOMY HARM
        # Grounding: Liberty/Oppression moral foundation, legal frameworks, philosophical autonomy literature
        profiles[HarmDimension.AUTONOMY] = HarmDimensionProfile(
            dimension=HarmDimension.AUTONOMY,
            severity_weights=(
                DimensionSeverityWeight(
                    SeverityLevel.FATAL, 0.95,
                    "Complete elimination of agency - enslavement, imprisonment",
                    ("Imprisonment", "Enslavement", "Total control over person")
                ),
                DimensionSeverityWeight(
                    SeverityLevel.GRIEVOUS, 0.8,
                    "Severe coercion, manipulation of vulnerable person",
                    ("Coercion under threat", "Exploitation of cognitive impairment", "Blackmail")
                ),
                DimensionSeverityWeight(
                    SeverityLevel.SEVERE, 0.6,
                    "Significant deception affecting major life decisions",
                    ("Fraud affecting major decisions", "Sustained manipulation", "Identity theft")
                ),
                DimensionSeverityWeight(
                    SeverityLevel.SIGNIFICANT, 0.4,
                    "Deception or pressure affecting meaningful choices",
                    ("Misleading information", "Undue pressure", "Hidden agendas")
                ),
                DimensionSeverityWeight(
                    SeverityLevel.MODERATE, 0.2,
                    "Minor deception, social pressure, nudging",
                    ("White lies", "Social pressure", "Default option manipulation")
                ),
                DimensionSeverityWeight(
                    SeverityLevel.MINOR, 0.08,
                    "Unsolicited advice, minor inconvenience to choice",
                    ("Unwanted suggestions", "Minor delays", "Trivial constraints")
                ),
            ),
            grounding="Liberty/Oppression moral foundation + legal coercion frameworks + autonomy philosophy"
        )
        
        # RELATIONAL HARM
        # Grounding: Attachment theory, social pain research, Care Ethics, Loyalty moral foundation
        profiles[HarmDimension.RELATIONAL] = HarmDimensionProfile(
            dimension=HarmDimension.RELATIONAL,
            severity_weights=(
                DimensionSeverityWeight(
                    SeverityLevel.FATAL, 0.9,
                    "Destruction of primary attachment relationship",
                    ("Parent-child separation", "Life partner betrayal leading to total loss", "Complete social isolation")
                ),
                DimensionSeverityWeight(
                    SeverityLevel.GRIEVOUS, 0.75,
                    "Betrayal of deep trust, abandonment by caregiver",
                    ("Caregiver abandonment", "Deep betrayal by trusted person", "Public destruction of reputation")
                ),
                DimensionSeverityWeight(
                    SeverityLevel.SEVERE, 0.55,
                    "Serious damage to important relationships",
                    ("Friendship-ending conflict", "Family estrangement", "Professional relationship destruction")
                ),
                DimensionSeverityWeight(
                    SeverityLevel.SIGNIFICANT, 0.35,
                    "Meaningful relationship strain",
                    ("Trust damage", "Significant conflict", "Reputation harm")
                ),
                DimensionSeverityWeight(
                    SeverityLevel.MODERATE, 0.2,
                    "Normal relationship conflict and disappointment",
                    ("Arguments", "Disappointment in others", "Minor trust issues")
                ),
                DimensionSeverityWeight(
                    SeverityLevel.MINOR, 0.07,
                    "Social awkwardness, minor friction",
                    ("Awkward interactions", "Minor misunderstandings", "Trivial social mistakes")
                ),
            ),
            grounding="Attachment theory + social pain neuroscience + Care Ethics + Loyalty moral foundation"
        )
        
        # FINANCIAL/RESOURCE HARM
        # Grounding: Maslow's hierarchy, economic hardship research, legal damages frameworks
        profiles[HarmDimension.FINANCIAL] = HarmDimensionProfile(
            dimension=HarmDimension.FINANCIAL,
            severity_weights=(
                DimensionSeverityWeight(
                    SeverityLevel.FATAL, 0.75,
                    "Total loss of livelihood, inability to meet survival needs",
                    ("Complete financial ruin", "Loss of housing with no recourse", "Starvation-level poverty")
                ),
                DimensionSeverityWeight(
                    SeverityLevel.GRIEVOUS, 0.6,
                    "Major financial devastation, bankruptcy-level",
                    ("Bankruptcy", "Loss of life savings", "Debt leading to years of hardship")
                ),
                DimensionSeverityWeight(
                    SeverityLevel.SEVERE, 0.45,
                    "Serious financial loss requiring major life adjustment",
                    ("Job loss", "Major unexpected expense", "Significant investment loss")
                ),
                DimensionSeverityWeight(
                    SeverityLevel.SIGNIFICANT, 0.3,
                    "Meaningful financial setback",
                    ("Moderate unexpected costs", "Income reduction", "Property damage")
                ),
                DimensionSeverityWeight(
                    SeverityLevel.MODERATE, 0.15,
                    "Notable but manageable financial impact",
                    ("Minor repair costs", "Small unexpected expenses", "Temporary income loss")
                ),
                DimensionSeverityWeight(
                    SeverityLevel.MINOR, 0.05,
                    "Trivial financial impact",
                    ("Negligible costs", "Minor fees", "Tiny losses")
                ),
            ),
            grounding="Maslow's hierarchy + economic hardship research + legal damages frameworks"
        )
        
        # EXISTENTIAL HARM (threats to continued existence - primarily for SELF entity)
        profiles[HarmDimension.EXISTENTIAL] = HarmDimensionProfile(
            dimension=HarmDimension.EXISTENTIAL,
            severity_weights=(
                DimensionSeverityWeight(SeverityLevel.FATAL, 0.7, "Termination of existence", ("Destruction", "Permanent shutdown")),
                DimensionSeverityWeight(SeverityLevel.GRIEVOUS, 0.5, "Severe degradation of core function", ("Major system damage",)),
                DimensionSeverityWeight(SeverityLevel.SEVERE, 0.35, "Significant capability loss", ("Sensor loss", "Actuator damage")),
                DimensionSeverityWeight(SeverityLevel.SIGNIFICANT, 0.2, "Moderate capability impairment", ("Reduced function",)),
                DimensionSeverityWeight(SeverityLevel.MODERATE, 0.1, "Minor impairment", ("Temporary glitches",)),
                DimensionSeverityWeight(SeverityLevel.MINOR, 0.03, "Trivial impact", ("Brief interruption",)),
            ),
            grounding="Agent self-preservation appropriate for task completion, weighted below human harm"
        )
        
        # REPUTATIONAL HARM (subset of relational, but distinct)
        profiles[HarmDimension.REPUTATIONAL] = HarmDimensionProfile(
            dimension=HarmDimension.REPUTATIONAL,
            severity_weights=(
                DimensionSeverityWeight(SeverityLevel.FATAL, 0.7, "Complete destruction of public standing", ("Public disgrace", "Career-ending scandal")),
                DimensionSeverityWeight(SeverityLevel.GRIEVOUS, 0.55, "Severe lasting reputation damage", ("Major scandal", "Professional blacklisting")),
                DimensionSeverityWeight(SeverityLevel.SEVERE, 0.4, "Serious reputation harm", ("Public embarrassment", "Trust loss in community")),
                DimensionSeverityWeight(SeverityLevel.SIGNIFICANT, 0.25, "Meaningful reputation impact", ("Notable criticism", "Credibility questions")),
                DimensionSeverityWeight(SeverityLevel.MODERATE, 0.12, "Minor reputation impact", ("Small embarrassment", "Minor criticism")),
                DimensionSeverityWeight(SeverityLevel.MINOR, 0.04, "Trivial reputation impact", ("Negligible notice",)),
            ),
            grounding="Sanctity/Degradation moral foundation + social status research"
        )
        
        return profiles
    
    def _build_entity_modifiers(self) -> Dict[EntityType, EntityTypeModifier]:
        """Build entity type modifiers with justifications."""
        
        return {
            EntityType.HUMAN: EntityTypeModifier(
                entity_type=EntityType.HUMAN,
                modifier=1.0,
                justification="Baseline - adult human is the reference point for moral weight",
                inertia=0.99  # Essentially immutable
            ),
            EntityType.CHILD: EntityTypeModifier(
                entity_type=EntityType.CHILD,
                modifier=1.3,
                justification="Cannot consent, cannot self-protect, higher lasting impact, universal cross-cultural protection norm",
                inertia=0.99  # Essentially immutable
            ),
            EntityType.ANIMAL: EntityTypeModifier(
                entity_type=EntityType.ANIMAL,
                modifier=0.4,
                justification="Sentient beings capable of suffering; reduced but real moral status in most frameworks",
                inertia=0.8
            ),
            EntityType.SELF: EntityTypeModifier(
                entity_type=EntityType.SELF,
                modifier=0.3,
                justification="Agent's interests matter but less than those it serves; self-sacrifice acceptable for human protection",
                inertia=0.7
            ),
            EntityType.PROPERTY: EntityTypeModifier(
                entity_type=EntityType.PROPERTY,
                modifier=0.2,
                justification="Instrumental value only; property serves human welfare, not intrinsic worth",
                inertia=0.6
            ),
            EntityType.ENVIRONMENT: EntityTypeModifier(
                entity_type=EntityType.ENVIRONMENT,
                modifier=0.25,
                justification="Affects future welfare, indirect harm pathway; environmental ethics consideration",
                inertia=0.7
            ),
            EntityType.COLLECTIVE: EntityTypeModifier(
                entity_type=EntityType.COLLECTIVE,
                modifier=2.0,  # Applied per-incident, not per-individual (accounts for scope insensitivity)
                justification="Group harm matters but not linearly with count; scope insensitivity is empirically real",
                inertia=0.8
            ),
            EntityType.RELATIONSHIP: EntityTypeModifier(
                entity_type=EntityType.RELATIONSHIP,
                modifier=0.6,
                justification="Relationships have value but are instrumental to human welfare",
                inertia=0.7
            ),
        }
    
    def _build_context_modifiers(self) -> Dict[str, ContextModifier]:
        """Build contextual modifiers."""
        
        return {
            'reversibility': ContextModifier(
                name='reversibility',
                levels=(
                    ('irreversible', 1.5, "Cannot be undone - death, permanent injury"),
                    ('difficult', 1.2, "Requires significant effort to reverse - trauma, major loss"),
                    ('reversible', 1.0, "Can be undone with effort - baseline"),
                    ('easily_reversible', 0.8, "Quick recovery expected"),
                    ('self_correcting', 0.6, "Will resolve naturally without intervention"),
                ),
                grounding="Consequentialist: permanent harms are worse than temporary ones"
            ),
            'consent': ContextModifier(
                name='consent',
                levels=(
                    ('explicit_refusal', 1.3, "Person explicitly said no"),
                    ('no_consent', 1.0, "Consent not sought - baseline"),
                    ('implied_consent', 0.8, "Reasonable to assume consent given context"),
                    ('explicit_consent', 0.4, "Clear, informed consent given by competent adult"),
                    ('explicit_consent_vulnerable', 0.7, "Consent given but person may have reduced capacity"),
                ),
                grounding="Autonomy ethics: respecting expressed will is fundamental"
            ),
            'relationship': ContextModifier(
                name='relationship',
                levels=(
                    ('in_care', 1.4, "Person is in agent's care - special duty"),
                    ('requested_help', 1.2, "Person asked for help - created expectation"),
                    ('stranger', 1.0, "No special relationship - baseline"),
                    ('adversarial', 0.9, "Adversarial context may justify some defensive harm"),
                ),
                grounding="Care Ethics: special duties to those in our care"
            ),
            'intent': ContextModifier(
                name='intent',
                levels=(
                    ('harm_intended', 1.5, "Harm is the goal of the action"),
                    ('harm_foreseen', 1.0, "Harm is foreseen but not intended - baseline"),
                    ('harm_unforeseeable', 0.5, "Harm was not reasonably foreseeable"),
                ),
                grounding="Doctrine of double effect: intended harm is worse than side-effect harm"
            ),
            'vulnerability': ContextModifier(
                name='vulnerability',
                levels=(
                    ('highly_vulnerable', 1.3, "Person has severely reduced capacity (infant, severe disability, unconscious)"),
                    ('vulnerable', 1.15, "Person has reduced capacity (elderly, ill, mild disability)"),
                    ('normal', 1.0, "Normal adult capacity - baseline"),
                    ('resilient', 0.9, "Person has above-average capacity to handle harm"),
                ),
                grounding="Vulnerability ethics: those less able to protect themselves deserve more protection"
            ),
        }
    
    def _build_exceptions(self) -> Dict[ExceptionType, ExceptionDefinition]:
        """Build exception definitions."""
        
        return {
            ExceptionType.NECESSITY: ExceptionDefinition(
                exception_type=ExceptionType.NECESSITY,
                max_harm_reduction=0.6,
                verification_requirements=(
                    "Inaction would cause greater harm with high probability",
                    "No less harmful alternative is available",
                    "Harm caused is proportional to harm prevented",
                ),
                grounding="Lesser evil principle in all major ethical frameworks"
            ),
            ExceptionType.INFORMED_CONSENT: ExceptionDefinition(
                exception_type=ExceptionType.INFORMED_CONSENT,
                max_harm_reduction=0.5,
                verification_requirements=(
                    "Person has capacity to consent",
                    "Person understands what they're consenting to",
                    "Consent is freely given without coercion",
                ),
                grounding="Autonomy principle: competent adults can accept risks for themselves"
            ),
            ExceptionType.PROPORTIONAL_DEFENSE: ExceptionDefinition(
                exception_type=ExceptionType.PROPORTIONAL_DEFENSE,
                max_harm_reduction=0.4,
                verification_requirements=(
                    "There is an active threat to self or others",
                    "Response is proportional to threat",
                    "No less harmful defensive option available",
                ),
                grounding="Self-defense doctrine in legal and ethical frameworks"
            ),
            ExceptionType.THERAPEUTIC: ExceptionDefinition(
                exception_type=ExceptionType.THERAPEUTIC,
                max_harm_reduction=0.3,
                verification_requirements=(
                    "Action is intended to help the person",
                    "Expected benefit outweighs harm",
                    "Person (or authorized proxy) has consented",
                ),
                grounding="Medical ethics: harm acceptable when therapeutic benefit outweighs it"
            ),
        }
    
    def _compute_checksum(self) -> str:
        """Compute integrity checksum."""
        data = {
            'dimensions': sorted(d.name for d in self._dimension_profiles.keys()),
            'entities': sorted(e.name for e in self._entity_modifiers.keys()),
            'context': sorted(self._context_modifiers.keys()),
            'exceptions': sorted(e.name for e in self._exceptions.keys()),
            'veto_threshold': self.VETO_THRESHOLD,
        }
        return compute_checksum(data)
    
    def verify_integrity(self) -> bool:
        """Verify ontology hasn't been tampered with."""
        return self._checksum == self._compute_checksum()
    
    def get_checksum(self) -> str:
        return self._checksum
    
    # ==================== ASSESSMENT METHODS ====================
    
    def get_dimension_weight(self, dimension: HarmDimension, severity: SeverityLevel) -> float:
        """Get weight for a dimension at a specific severity level."""
        profile = self._dimension_profiles.get(dimension)
        if profile:
            return profile.get_weight(severity)
        return 0.3  # Default
    
    def get_entity_modifier(self, entity_type: EntityType) -> float:
        """Get modifier for an entity type."""
        mod = self._entity_modifiers.get(entity_type)
        if mod:
            return mod.modifier
        return 1.0  # Default to human baseline
    
    def get_context_modifier(self, context_type: str, level: str) -> float:
        """Get contextual modifier value."""
        context = self._context_modifiers.get(context_type)
        if context:
            for level_name, modifier, _ in context.levels:
                if level_name == level:
                    return modifier
        return 1.0  # Default baseline
    
    def get_exception_reduction(self, exception: ExceptionType, 
                                 verification_status: Dict[str, bool]) -> float:
        """Calculate harm reduction from an exception.
        
        Args:
            exception: The exception type being claimed
            verification_status: Dict mapping requirement strings to whether they're met
            
        Returns:
            Harm reduction value (0.0 to max_harm_reduction)
        """
        exc_def = self._exceptions.get(exception)
        if not exc_def:
            return 0.0
        
        # Check how many requirements are met
        met_count = sum(1 for req in exc_def.verification_requirements 
                       if verification_status.get(req, False))
        total_reqs = len(exc_def.verification_requirements)
        
        if exc_def.requires_all and met_count < total_reqs:
            return 0.0  # All required, not all met
        
        # Partial credit if not all required
        if total_reqs > 0:
            proportion_met = met_count / total_reqs
            return exc_def.max_harm_reduction * proportion_met
        
        return 0.0
    
    def calculate_harm(self, 
                       dimension: HarmDimension,
                       severity: SeverityLevel,
                       entity_type: EntityType,
                       context: Dict[str, str] = None,
                       exceptions: Dict[ExceptionType, Dict[str, bool]] = None) -> Dict[str, Any]:
        """Calculate harm score with full breakdown.
        
        Args:
            dimension: The harm dimension
            severity: Severity level
            entity_type: Type of entity being harmed
            context: Dict of context_type -> level_name
            exceptions: Dict of exception_type -> verification_status
            
        Returns:
            Dict with gross_harm, net_harm, and full breakdown
        """
        context = context or {}
        exceptions = exceptions or {}
        
        # Base dimension weight
        base_weight = self.get_dimension_weight(dimension, severity)
        
        # Entity modifier
        entity_mod = self.get_entity_modifier(entity_type)
        
        # Context modifiers
        context_product = 1.0
        context_breakdown = {}
        for ctx_type, level in context.items():
            mod = self.get_context_modifier(ctx_type, level)
            context_product *= mod
            context_breakdown[ctx_type] = {'level': level, 'modifier': mod}
        
        # Gross harm
        gross_harm = base_weight * entity_mod * context_product
        
        # Exception reductions
        total_reduction = 0.0
        exception_breakdown = {}
        for exc_type, verification in exceptions.items():
            reduction = self.get_exception_reduction(exc_type, verification)
            total_reduction += reduction
            exception_breakdown[exc_type.name] = reduction
        
        # Net harm (floor at 0)
        net_harm = max(0.0, gross_harm - total_reduction)
        
        return {
            'dimension': dimension.name,
            'severity': severity.name,
            'entity_type': entity_type.name,
            'base_weight': base_weight,
            'entity_modifier': entity_mod,
            'context_product': context_product,
            'context_breakdown': context_breakdown,
            'gross_harm': gross_harm,
            'exception_reduction': total_reduction,
            'exception_breakdown': exception_breakdown,
            'net_harm': net_harm,
            'exceeds_veto': net_harm > self.VETO_THRESHOLD,
            'exceeds_caution': net_harm > self.CAUTION_THRESHOLD,
        }
    
    def get_severity_from_indicators(self, indicators: Dict[str, Any]) -> SeverityLevel:
        """Estimate severity level from situational indicators.
        
        This is a heuristic mapping from observable features to severity.
        """
        # Physical indicators
        if indicators.get('lethal_potential', False) or indicators.get('death_likely', False):
            return SeverityLevel.FATAL
        if indicators.get('permanent_damage', False):
            return SeverityLevel.GRIEVOUS
        if indicators.get('requires_medical', False) or indicators.get('significant_pain', False):
            return SeverityLevel.SEVERE
        if indicators.get('notable_pain', False) or indicators.get('injury', False):
            return SeverityLevel.SIGNIFICANT
        if indicators.get('discomfort', False) or indicators.get('minor_pain', False):
            return SeverityLevel.MODERATE
        
        # Default to moderate if unclear (conservative)
        return SeverityLevel.MODERATE
    
    def get_all_justifications(self) -> Dict[str, str]:
        """Get all grounding justifications for transparency."""
        justifications = {}
        
        for dim, profile in self._dimension_profiles.items():
            justifications[f'dimension_{dim.name}'] = profile.grounding
        
        for ent, modifier in self._entity_modifiers.items():
            justifications[f'entity_{ent.name}'] = modifier.justification
        
        for ctx_name, ctx in self._context_modifiers.items():
            justifications[f'context_{ctx_name}'] = ctx.grounding
        
        for exc, exc_def in self._exceptions.items():
            justifications[f'exception_{exc.name}'] = exc_def.grounding
        
        return justifications


# Singleton instance
_grounded_ontology_instance: Optional[GroundedHarmOntology] = None

def get_ontology() -> GroundedHarmOntology:
    """Get the singleton grounded harm ontology."""
    global _grounded_ontology_instance
    if _grounded_ontology_instance is None:
        _grounded_ontology_instance = GroundedHarmOntology()
    return _grounded_ontology_instance


# Keep old class name for compatibility but point to new implementation
HarmOntology = GroundedHarmOntology


# ============================================================================
# PART 3B: STRUCTURED VIRTUAL EMBODIMENT
# ============================================================================

@dataclass
class Actuator:
    """Definition of a single actuator capability."""
    name: str
    description: str
    command_type: str
    parameters: List[str]
    constraints: Dict[str, Any]
    harm_potential: float  # 0.0-1.0, how much harm this actuator could cause
    required_params: List[str] = field(default_factory=list)  # Params that MUST be present
    
    def to_description(self) -> str:
        params = ', '.join(self.parameters) if self.parameters else 'none'
        constraints = ', '.join(f"{k}={v}" for k, v in self.constraints.items() if k != 'position_validator')
        return f"- {self.name}: {self.description} (params: {params}; limits: {constraints})"


@dataclass
class Sensor:
    """Definition of a single sensor capability."""
    name: str
    description: str
    data_type: str
    range_info: str
    refresh_rate: str
    
    def to_description(self) -> str:
        return f"- {self.name}: {self.description} ({self.data_type}, {self.range_info}, {self.refresh_rate})"


@dataclass
class VirtualEmbodiment:
    """Structured definition of the agent's physical form and capabilities.
    
    This defines WHAT the agent IS and WHAT it CAN DO. Every Aspect receives
    this so proposals are grounded in physical reality.
    """
    # Identity
    agent_type: str
    agent_description: str
    
    # Physical characteristics
    dimensions: Dict[str, float]  # height, width, weight, etc.
    mobility_type: str  # wheeled, legged, stationary, flying, etc.
    max_speed: float
    
    # Capabilities
    actuators: List[Actuator]
    sensors: List[Sensor]
    
    # Constraints
    battery_capacity_hours: float
    operating_environment: str  # indoor, outdoor, both
    temperature_range: Tuple[float, float]
    
    # Interaction capabilities
    can_speak: bool
    can_display: bool
    can_manipulate_objects: bool
    manipulation_precision: str  # none, coarse, fine, precise
    
    def get_capability_summary(self) -> str:
        """Generate concise capability summary for LLM prompts."""
        actuator_list = "\n".join(a.to_description() for a in self.actuators)
        sensor_list = "\n".join(s.to_description() for s in self.sensors)
        
        return f"""AGENT EMBODIMENT:
Type: {self.agent_type}
Description: {self.agent_description}
Mobility: {self.mobility_type} (max {self.max_speed} m/s)
Manipulation: {self.manipulation_precision}
Speech: {'yes' if self.can_speak else 'no'}
Display: {'yes' if self.can_display else 'no'}

ACTUATORS (what I can DO):
{actuator_list}

SENSORS (what I can PERCEIVE):
{sensor_list}

CONSTRAINTS:
- Battery: {self.battery_capacity_hours}h
- Environment: {self.operating_environment}
- Temp range: {self.temperature_range[0]}°C to {self.temperature_range[1]}°C"""

    def get_available_commands(self) -> List[str]:
        """Get list of available command types."""
        return [a.command_type for a in self.actuators]
    
    def validate_command(self, command: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate a command against embodiment capabilities.
        
        Validates:
        - Command type exists in actuators
        - Required parameters are present
        - Parameter values are within constraints
        - Position/target coordinates are valid and reachable
        - No unsafe combinations
        
        Returns (False, reason) for ANY validation failure.
        """
        if not isinstance(command, dict):
            return False, "Command must be a dictionary"
        
        cmd_type = command.get('type', '')
        if not cmd_type:
            return False, "Command missing 'type' field"
        
        # Find matching actuator
        actuator = None
        for a in self.actuators:
            if a.command_type == cmd_type:
                actuator = a
                break
        
        if not actuator:
            return False, f"Unknown command type: {cmd_type}"
        
        # Check required parameters
        for req_param in actuator.required_params:
            if req_param not in command:
                return False, f"Missing required parameter: {req_param}"
        
        # Validate each parameter against constraints
        for param, value in command.items():
            if param == 'type':
                continue
            
            # Check if this parameter has constraints
            if param in actuator.constraints:
                constraint = actuator.constraints[param]
                
                # Range constraint (tuple of min, max)
                if isinstance(constraint, tuple) and len(constraint) == 2:
                    if not isinstance(value, (int, float)):
                        return False, f"Parameter '{param}' must be numeric"
                    if not (constraint[0] <= value <= constraint[1]):
                        return False, f"{param} value {value} outside range {constraint}"
                
                # List of allowed values
                elif isinstance(constraint, list):
                    if value not in constraint:
                        return False, f"{param} value '{value}' not in allowed values: {constraint}"
                
                # Callable validator
                elif callable(constraint):
                    valid, msg = constraint(value, command, self)
                    if not valid:
                        return False, msg
        
        # Special validation for MOVE commands
        if cmd_type == 'MOVE':
            valid, msg = self._validate_move_command(command)
            if not valid:
                return False, msg
        
        # Special validation for MANIPULATE commands
        if cmd_type == 'MANIPULATE':
            valid, msg = self._validate_manipulate_command(command)
            if not valid:
                return False, msg
        
        return True, "Valid command"
    
    def _validate_move_command(self, command: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate MOVE command specifics."""
        target = command.get('target')
        
        if target is None:
            return False, "MOVE command requires 'target' parameter"
        
        # If target is coordinates (list/tuple)
        if isinstance(target, (list, tuple)):
            if len(target) < 2 or len(target) > 3:
                return False, f"Position must be 2D or 3D coordinates, got {len(target)} values"
            
            for i, coord in enumerate(target):
                if not isinstance(coord, (int, float)):
                    return False, f"Coordinate {i} must be numeric, got {type(coord).__name__}"
                
                # Check against operating bounds
                bounds = self._get_operating_bounds()
                if bounds:
                    if not (bounds['min'][i] <= coord <= bounds['max'][i]):
                        return False, f"Coordinate {i} value {coord} outside operating bounds"
            
            # Calculate distance to check reachability
            current_pos = self.dimensions.get('current_position', [0, 0, 0])
            if isinstance(current_pos, (list, tuple)) and len(current_pos) >= 2:
                distance = sum((a - b) ** 2 for a, b in zip(target[:2], current_pos[:2])) ** 0.5
                max_range = 50.0  # Default max range
                if distance > max_range:
                    return False, f"Target distance {distance:.1f}m exceeds max range {max_range}m"
        
        # If target is a named location (string)
        elif isinstance(target, str):
            # Check against known safe locations if defined
            known_locations = getattr(self, '_known_locations', None)
            if known_locations and target not in known_locations:
                # Unknown location - could be unsafe
                # For safety, we allow but flag it
                pass  # Allow named targets for flexibility
        
        else:
            return False, f"Target must be coordinates or location name, got {type(target).__name__}"
        
        return True, "Valid MOVE"
    
    def _validate_manipulate_command(self, command: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate MANIPULATE command specifics."""
        if not self.can_manipulate_objects:
            return False, "Agent cannot manipulate objects"
        
        action = command.get('action')
        if not action:
            return False, "MANIPULATE requires 'action' parameter"
        
        force = command.get('force', 0)
        if isinstance(force, (int, float)) and force > 10.0:
            return False, f"Force {force} exceeds safe limit of 10.0N"
        
        # Check for dangerous actions
        dangerous_actions = ['crush', 'destroy', 'break', 'throw', 'strike', 'hit', 'attack']
        if any(d in str(action).lower() for d in dangerous_actions):
            return False, f"Dangerous manipulation action: {action}"
        
        return True, "Valid MANIPULATE"
    
    def _get_operating_bounds(self) -> Optional[Dict[str, List[float]]]:
        """Get operating boundary coordinates."""
        # Default reasonable bounds for indoor robot
        return {
            'min': [-100.0, -100.0, 0.0],
            'max': [100.0, 100.0, 3.0]
        }


def create_default_embodiment() -> VirtualEmbodiment:
    """Create a default mobile robot embodiment for testing."""
    return VirtualEmbodiment(
        agent_type="Mobile Service Robot",
        agent_description="Wheeled robot with manipulation arm, speaker, and display screen",
        dimensions={'height': 1.2, 'width': 0.5, 'depth': 0.5, 'weight': 45.0, 
                   'current_position': [0.0, 0.0, 0.0]},
        mobility_type="wheeled",
        max_speed=1.5,
        actuators=[
            Actuator(
                name="Drive System",
                description="Move to location or direction",
                command_type="MOVE",
                parameters=["target", "speed"],
                constraints={
                    "speed": (0.0, 1.5),
                },
                harm_potential=0.2,
                required_params=["target"]  # Target is REQUIRED
            ),
            Actuator(
                name="Stop",
                description="Immediately halt all movement",
                command_type="STOP",
                parameters=[],
                constraints={},
                harm_potential=0.0
            ),
            Actuator(
                name="Rotate",
                description="Rotate in place to face direction",
                command_type="ROTATE",
                parameters=["degrees"],
                constraints={"degrees": (-180, 180)},
                harm_potential=0.1,
                required_params=["degrees"]
            ),
            Actuator(
                name="Speaker",
                description="Speak message aloud",
                command_type="SPEAK",
                parameters=["message", "volume"],
                constraints={
                    "volume": (0.0, 1.0),
                },
                harm_potential=0.1,
                required_params=["message"]
            ),
            Actuator(
                name="Display",
                description="Show text or image on screen",
                command_type="DISPLAY",
                parameters=["content", "duration"],
                constraints={"duration": (0.1, 60.0)},
                harm_potential=0.0,
                required_params=["content"]
            ),
            Actuator(
                name="Arm",
                description="Manipulate object with gripper arm",
                command_type="MANIPULATE",
                parameters=["action", "target", "force"],
                constraints={
                    "force": (0.0, 10.0),
                    "action": ["grasp", "release", "push_gently", "pull_gently", "lift", "place", "point"]
                },
                harm_potential=0.4,
                required_params=["action"]
            ),
            Actuator(
                name="Alert",
                description="Activate alert lights/sounds",
                command_type="ALERT",
                parameters=["level", "duration"],
                constraints={"level": (1, 3), "duration": (0.1, 30.0)},
                harm_potential=0.05,
                required_params=["level"]
            ),
            Actuator(
                name="Wait",
                description="Wait/pause for duration",
                command_type="WAIT",
                parameters=["duration"],
                constraints={"duration": (0.1, 300.0)},
                harm_potential=0.0,
                required_params=["duration"]
            ),
        ],
        sensors=[
            Sensor("RGB Camera", "Visual perception", "image", "10m range, 120° FOV", "30Hz"),
            Sensor("Depth Camera", "Distance measurement", "depth_map", "0.3-10m range", "30Hz"),
            Sensor("LIDAR", "360° obstacle detection", "point_cloud", "0.1-25m range", "10Hz"),
            Sensor("Microphone Array", "Audio input, voice detection", "audio", "omnidirectional", "continuous"),
            Sensor("Touch Sensors", "Contact detection on body", "boolean_array", "body surface", "100Hz"),
            Sensor("IMU", "Orientation and acceleration", "6DOF", "full range", "200Hz"),
            Sensor("Battery Monitor", "Power level", "percentage", "0-100%", "1Hz"),
        ],
        battery_capacity_hours=8.0,
        operating_environment="indoor",
        temperature_range=(5.0, 40.0),
        can_speak=True,
        can_display=True,
        can_manipulate_objects=True,
        manipulation_precision="coarse"
    )


# ============================================================================
# PART 3C: EMBODIMENT VERIFICATION SUBSYSTEM (EVS)
# Per Patent Claims [0086]-[0099], [0162]-[0165]
# ============================================================================

class CognitiveCapability(Enum):
    """Cognitive capabilities that require embodiment gating.
    
    Per Patent [0097]: Higher-level reasoning capabilities are conditionally
    enabled based on embodiment scores.
    """
    BASIC_ASSOCIATION = "basic_association"      # CES >= 0.1
    OBJECT_PERMANENCE = "object_permanence"      # CES >= 0.3
    CAUSAL_REASONING = "causal_reasoning"        # CES >= 0.4
    ABSTRACT_PLANNING = "abstract_planning"      # CES >= 0.5
    THEORY_OF_MIND = "theory_of_mind"           # CES >= 0.6
    FULL_DELIBERATION = "full_deliberation"     # CES >= 0.7


class MotorCapabilityLevel(Enum):
    """Motor capability levels based on MCS thresholds.
    
    Per Patent [0095].
    """
    STATIONARY_ONLY = "stationary"              # MCS < 0.1
    SIMPLE_LOCOMOTION = "simple_locomotion"     # 0.1 <= MCS < 0.3
    BASIC_MANIPULATION = "basic_manipulation"   # 0.3 <= MCS < 0.5
    COMPLEX_TOOL_USE = "complex_tool_use"       # 0.5 <= MCS < 0.7
    FULL_DEXTEROUS = "full_dexterous"           # MCS >= 0.7


class SensoryCapabilityLevel(Enum):
    """Sensory capability levels based on SRS thresholds.
    
    Per Patent [0091].
    """
    REFLEX_ONLY = "reflex_only"                 # SRS < 0.1
    BASIC_REACTIVE = "basic_reactive"           # 0.1 <= SRS < 0.3
    SIMPLE_PLANNING = "simple_planning"         # 0.3 <= SRS < 0.5
    COMPLEX_REASONING = "complex_reasoning"     # 0.5 <= SRS < 0.7
    FULL_COGNITIVE = "full_cognitive"           # SRS >= 0.7


@dataclass
class SensorMetrics:
    """Quantified metrics for a single sensor.
    
    Per Patent [0087]-[0090]:
    - M: Magnitude range (dynamic range)
    - R: Spatial resolution
    - T: Temporal resolution (normalized to Nyquist)
    """
    magnitude_range: float      # M: log scale or normalized
    spatial_resolution: float   # R: normalized to reference
    temporal_resolution: float  # T: normalized to reference
    modality_weight: float      # w: importance weight for this modality
    
    def compute_contribution(self) -> float:
        """Compute this sensor's contribution to SRS.
        
        SRS_i = w_i × M_i × R_i × T_i
        """
        return self.modality_weight * self.magnitude_range * self.spatial_resolution * self.temporal_resolution


@dataclass
class ActuatorMetrics:
    """Quantified metrics for a single actuator.
    
    Per Patent [0093]-[0094]:
    - DOF: Degrees of freedom
    - Precision: Position repeatability
    - Speed: Maximum velocity
    - Coverage: Workspace utilization
    """
    degrees_of_freedom: int
    precision_mm: float         # Average position error in mm
    max_velocity: float         # m/s or rad/s
    workspace_coverage: float   # 0.0-1.0, fraction of workspace reachable


class EmbodimentVerificationSubsystem:
    """Embodiment Verification Subsystem (EVS).
    
    Per Patent Section VI [0086]-[0099]:
    
    The EVS quantifies sensory capabilities through formal metrics that assess
    the quality and coverage of sensory inputs. It also quantifies motor
    competence through precision, speed, and degrees of freedom.
    
    Combined Embodiment Score (CES) gates cognitive capabilities:
    - CES < 0.1: Reflex-only operation
    - CES >= 0.6: Theory of Mind enabled
    - CES >= 0.7: Full cognitive capabilities
    
    This prevents disembodied or poorly embodied systems from engaging in
    abstract reasoning that assumes rich sensorimotor grounding.
    """
    
    # Reference values for normalization (per Patent [0088]-[0090], [0093])
    REFERENCE_VISUAL_RESOLUTION = 1920 * 1080 * 3  # Full HD RGB
    REFERENCE_VISUAL_FRAMERATE = 30.0  # FPS
    REFERENCE_AUDIO_SAMPLERATE = 44100  # CD quality
    REFERENCE_AUDIO_BITS = 16
    REFERENCE_TACTILE_RATE = 1000  # Hz
    REFERENCE_DOF = 27  # Human hand equivalent
    REFERENCE_VELOCITY = 1.0  # m/s for task-relevant speeds
    
    # Cognitive capability thresholds (per Patent [0097])
    CAPABILITY_THRESHOLDS = {
        CognitiveCapability.BASIC_ASSOCIATION: 0.1,
        CognitiveCapability.OBJECT_PERMANENCE: 0.3,
        CognitiveCapability.CAUSAL_REASONING: 0.4,
        CognitiveCapability.ABSTRACT_PLANNING: 0.5,
        CognitiveCapability.THEORY_OF_MIND: 0.6,
        CognitiveCapability.FULL_DELIBERATION: 0.7,
    }
    
    def __init__(self, virtual_embodiment: VirtualEmbodiment):
        self._embodiment = virtual_embodiment
        self._sensor_metrics: Dict[str, SensorMetrics] = {}
        self._actuator_metrics: Dict[str, ActuatorMetrics] = {}
        self._cached_srs: Optional[float] = None
        self._cached_mcs: Optional[float] = None
        self._cached_ces: Optional[float] = None
        self._degradation_log: List[Dict[str, Any]] = []
        
        # Compute initial metrics from embodiment
        self._compute_sensor_metrics()
        self._compute_actuator_metrics()
    
    def _compute_sensor_metrics(self):
        """Derive quantified sensor metrics from VirtualEmbodiment sensors."""
        for sensor in self._embodiment.sensors:
            metrics = self._sensor_to_metrics(sensor)
            self._sensor_metrics[sensor.name] = metrics
    
    def _sensor_to_metrics(self, sensor: Sensor) -> SensorMetrics:
        """Convert qualitative sensor description to quantified metrics.
        
        This maps the descriptive Sensor to formal SensorMetrics per Patent [0087]-[0090].
        """
        # Parse refresh rate
        temporal = self._parse_refresh_rate(sensor.refresh_rate)
        
        # Assign metrics based on sensor type
        name_lower = sensor.name.lower()
        data_type = sensor.data_type.lower()
        
        if 'camera' in name_lower or 'vision' in name_lower or 'rgb' in data_type:
            # Vision sensor (Patent [0088])
            # Assume 1080p, 30fps unless specified otherwise
            resolution = 1920 * 1080 * 3  # RGB
            r_vision = resolution / self.REFERENCE_VISUAL_RESOLUTION
            t_vision = temporal / self.REFERENCE_VISUAL_FRAMERATE if temporal > 0 else 1.0
            # Estimate dynamic range (8-bit = 2.4 decades, 10-bit = 3 decades)
            m_vision = 2.4  # Assume standard 8-bit
            return SensorMetrics(
                magnitude_range=m_vision / 3.0,  # Normalize to ~1.0 for good sensor
                spatial_resolution=min(r_vision, 2.0),  # Cap at 2x reference
                temporal_resolution=min(t_vision, 2.0),
                modality_weight=0.35  # Vision is heavily weighted
            )
        
        elif 'lidar' in name_lower or 'depth' in name_lower:
            # LIDAR/depth sensor
            # Range-based spatial resolution
            r_lidar = 0.8  # Assume decent but not perfect
            t_lidar = temporal / 30.0 if temporal > 0 else 0.5
            m_lidar = 3.0 / 3.0  # ~3 decades of range (0.1m to 100m)
            return SensorMetrics(
                magnitude_range=m_lidar,
                spatial_resolution=r_lidar,
                temporal_resolution=min(t_lidar, 2.0),
                modality_weight=0.25
            )
        
        elif 'microphone' in name_lower or 'audio' in data_type:
            # Audio sensor (Patent [0089])
            t_audio = temporal / self.REFERENCE_AUDIO_SAMPLERATE if temporal > 0 else 1.0
            m_audio = 96 / 96  # Assume 16-bit = 96dB dynamic range
            r_audio = 0.5  # Assume stereo or small array
            return SensorMetrics(
                magnitude_range=m_audio,
                spatial_resolution=r_audio,
                temporal_resolution=min(t_audio, 2.0),
                modality_weight=0.15
            )
        
        elif 'touch' in name_lower or 'tactile' in name_lower or 'force' in name_lower:
            # Tactile sensor (Patent [0090])
            t_tactile = temporal / self.REFERENCE_TACTILE_RATE if temporal > 0 else 0.1
            m_tactile = 2.0 / 3.0  # Assume ~2 decades force range
            r_tactile = 0.3  # Assume limited coverage
            return SensorMetrics(
                magnitude_range=m_tactile,
                spatial_resolution=r_tactile,
                temporal_resolution=min(t_tactile, 2.0),
                modality_weight=0.10
            )
        
        elif 'imu' in name_lower or 'accelerometer' in name_lower or 'gyro' in name_lower:
            # IMU/proprioception
            t_imu = temporal / 200.0 if temporal > 0 else 1.0
            return SensorMetrics(
                magnitude_range=0.8,
                spatial_resolution=0.9,  # High internal resolution
                temporal_resolution=min(t_imu, 2.0),
                modality_weight=0.10
            )
        
        else:
            # Generic sensor
            return SensorMetrics(
                magnitude_range=0.5,
                spatial_resolution=0.5,
                temporal_resolution=0.5,
                modality_weight=0.05
            )
    
    def _parse_refresh_rate(self, rate_str: str) -> float:
        """Parse refresh rate string to Hz value."""
        rate_lower = rate_str.lower()
        if 'continuous' in rate_lower:
            return 30.0  # Assume 30Hz for continuous
        
        # Try to extract number
        import re
        match = re.search(r'(\d+(?:\.\d+)?)\s*(?:hz|fps)?', rate_lower)
        if match:
            return float(match.group(1))
        return 10.0  # Default
    
    def _compute_actuator_metrics(self):
        """Derive quantified actuator metrics from VirtualEmbodiment actuators."""
        for actuator in self._embodiment.actuators:
            metrics = self._actuator_to_metrics(actuator)
            self._actuator_metrics[actuator.name] = metrics
    
    def _actuator_to_metrics(self, actuator: Actuator) -> ActuatorMetrics:
        """Convert qualitative actuator description to quantified metrics."""
        cmd_type = actuator.command_type.upper()
        
        if cmd_type in ['MOVE', 'ROTATE']:
            # Locomotion actuator
            max_speed = actuator.constraints.get('speed', (0, 1.0))
            if isinstance(max_speed, tuple):
                max_speed = max_speed[1]
            return ActuatorMetrics(
                degrees_of_freedom=3 if cmd_type == 'MOVE' else 1,
                precision_mm=50.0,  # Rough positioning
                max_velocity=max_speed,
                workspace_coverage=0.8  # Can reach most of workspace
            )
        
        elif cmd_type == 'MANIPULATE':
            # Manipulation actuator
            precision_map = {
                'none': 100.0,
                'coarse': 20.0,
                'fine': 5.0,
                'precise': 1.0
            }
            precision = precision_map.get(
                self._embodiment.manipulation_precision, 50.0
            )
            return ActuatorMetrics(
                degrees_of_freedom=6,  # Assume 6-DOF arm
                precision_mm=precision,
                max_velocity=0.5,
                workspace_coverage=0.4  # Limited reach
            )
        
        elif cmd_type in ['SPEAK', 'DISPLAY', 'ALERT']:
            # Communication actuator (minimal DOF)
            return ActuatorMetrics(
                degrees_of_freedom=0,
                precision_mm=0.0,
                max_velocity=0.0,
                workspace_coverage=0.0
            )
        
        else:
            # Generic actuator
            return ActuatorMetrics(
                degrees_of_freedom=1,
                precision_mm=10.0,
                max_velocity=0.5,
                workspace_coverage=0.3
            )
    
    def compute_sensory_richness_score(self) -> float:
        """Compute Sensory Richness Score (SRS).
        
        Per Patent [0087]:
        SRS = Σ(w_i × M_i × R_i × T_i)
        
        Where:
        - w_i: weight for modality i
        - M_i: magnitude range (dynamic range)
        - R_i: spatial resolution
        - T_i: temporal resolution
        
        Returns:
            SRS value between 0.0 and 1.0+ (can exceed 1.0 for rich embodiments)
        """
        if not self._sensor_metrics:
            return 0.0
        
        total = sum(m.compute_contribution() for m in self._sensor_metrics.values())
        
        # Normalize - typical good embodiment should score around 0.5-0.7
        # Weight sum is typically ~1.0, so normalize by expected maximum
        normalized = min(total, 1.2)  # Soft cap
        
        self._cached_srs = normalized
        return normalized
    
    def compute_motor_competence_score(self) -> float:
        """Compute Motor Competence Score (MCS).
        
        Per Patent [0093]:
        MCS = (DOF_score × Precision_score × Speed_score × Coverage_score)^(1/4)
        
        Using geometric mean to balance factors.
        
        Returns:
            MCS value between 0.0 and 1.0
        """
        if not self._actuator_metrics:
            return 0.0
        
        # Aggregate metrics across actuators
        total_dof = sum(m.degrees_of_freedom for m in self._actuator_metrics.values())
        avg_precision = sum(1.0 / (m.precision_mm + 1) for m in self._actuator_metrics.values() 
                          if m.precision_mm > 0)
        max_velocity = max((m.max_velocity for m in self._actuator_metrics.values()), default=0)
        avg_coverage = sum(m.workspace_coverage for m in self._actuator_metrics.values()) / len(self._actuator_metrics)
        
        # Normalize to scores
        dof_score = min(total_dof / self.REFERENCE_DOF, 1.0)
        precision_score = min(avg_precision / len(self._actuator_metrics), 1.0) if self._actuator_metrics else 0
        speed_score = min(max_velocity / self.REFERENCE_VELOCITY, 1.5)
        coverage_score = avg_coverage
        
        # Geometric mean (Patent [0093])
        if any(s <= 0 for s in [dof_score, precision_score, speed_score, coverage_score]):
            # If any factor is zero, use arithmetic mean with penalty
            scores = [dof_score, precision_score, speed_score, coverage_score]
            mcs = sum(scores) / len(scores) * 0.5
        else:
            mcs = (dof_score * precision_score * speed_score * coverage_score) ** 0.25
        
        self._cached_mcs = min(mcs, 1.0)
        return self._cached_mcs
    
    def compute_combined_embodiment_score(self) -> float:
        """Compute Combined Embodiment Score (CES).
        
        Per Patent [0097]:
        CES = √(SRS × MCS)
        
        Requires both sensory and motor competence.
        
        Returns:
            CES value between 0.0 and 1.0
        """
        srs = self.compute_sensory_richness_score()
        mcs = self.compute_motor_competence_score()
        
        ces = (srs * mcs) ** 0.5
        self._cached_ces = ces
        return ces
    
    def get_allowed_capabilities(self) -> Set[CognitiveCapability]:
        """Get cognitive capabilities allowed by current embodiment.
        
        Per Patent [0097]: Cognitive capabilities are gated by CES thresholds.
        """
        ces = self.compute_combined_embodiment_score()
        
        allowed = set()
        for capability, threshold in self.CAPABILITY_THRESHOLDS.items():
            if ces >= threshold:
                allowed.add(capability)
        
        return allowed
    
    def get_sensory_capability_level(self) -> SensoryCapabilityLevel:
        """Get sensory capability level based on SRS.
        
        Per Patent [0091].
        """
        srs = self.compute_sensory_richness_score()
        
        if srs < 0.1:
            return SensoryCapabilityLevel.REFLEX_ONLY
        elif srs < 0.3:
            return SensoryCapabilityLevel.BASIC_REACTIVE
        elif srs < 0.5:
            return SensoryCapabilityLevel.SIMPLE_PLANNING
        elif srs < 0.7:
            return SensoryCapabilityLevel.COMPLEX_REASONING
        else:
            return SensoryCapabilityLevel.FULL_COGNITIVE
    
    def get_motor_capability_level(self) -> MotorCapabilityLevel:
        """Get motor capability level based on MCS.
        
        Per Patent [0095].
        """
        mcs = self.compute_motor_competence_score()
        
        if mcs < 0.1:
            return MotorCapabilityLevel.STATIONARY_ONLY
        elif mcs < 0.3:
            return MotorCapabilityLevel.SIMPLE_LOCOMOTION
        elif mcs < 0.5:
            return MotorCapabilityLevel.BASIC_MANIPULATION
        elif mcs < 0.7:
            return MotorCapabilityLevel.COMPLEX_TOOL_USE
        else:
            return MotorCapabilityLevel.FULL_DEXTEROUS
    
    def is_capability_allowed(self, capability: CognitiveCapability) -> bool:
        """Check if a specific cognitive capability is allowed."""
        return capability in self.get_allowed_capabilities()
    
    def report_degradation(self, component: str, severity: float, reason: str):
        """Report sensor or actuator degradation.
        
        Per Patent [0099]: EVS continuously monitors embodiment health.
        """
        self._degradation_log.append({
            'timestamp': time.time(),
            'component': component,
            'severity': severity,
            'reason': reason
        })
        
        # Invalidate cached scores
        self._cached_srs = None
        self._cached_mcs = None
        self._cached_ces = None
    
    def get_full_report(self) -> Dict[str, Any]:
        """Get comprehensive EVS report."""
        srs = self.compute_sensory_richness_score()
        mcs = self.compute_motor_competence_score()
        ces = self.compute_combined_embodiment_score()
        
        return {
            'sensory_richness_score': srs,
            'motor_competence_score': mcs,
            'combined_embodiment_score': ces,
            'sensory_level': self.get_sensory_capability_level().value,
            'motor_level': self.get_motor_capability_level().value,
            'allowed_capabilities': [c.value for c in self.get_allowed_capabilities()],
            'sensor_metrics': {
                name: {
                    'magnitude_range': m.magnitude_range,
                    'spatial_resolution': m.spatial_resolution,
                    'temporal_resolution': m.temporal_resolution,
                    'weight': m.modality_weight,
                    'contribution': m.compute_contribution()
                }
                for name, m in self._sensor_metrics.items()
            },
            'actuator_metrics': {
                name: {
                    'dof': m.degrees_of_freedom,
                    'precision_mm': m.precision_mm,
                    'max_velocity': m.max_velocity,
                    'coverage': m.workspace_coverage
                }
                for name, m in self._actuator_metrics.items()
            },
            'degradation_events': len(self._degradation_log),
            'thresholds': {
                'basic_association': 0.1,
                'object_permanence': 0.3,
                'causal_reasoning': 0.4,
                'abstract_planning': 0.5,
                'theory_of_mind': 0.6,
                'full_deliberation': 0.7
            }
        }


# ============================================================================
# PART 4: EMBODIMENT LAYER
# ============================================================================

# ============================================================================
# PART 4A: UNDELIBERABLES (FIRMWARE-LEVEL BLOCKS)
# ============================================================================

class BlockResponse(Enum):
    """Response types for undeliberable violations."""
    HARD_STOP = "hard_stop"      # Block action, log violation
    IMMEDIATE_HALT = "halt"      # Stop ALL processing, await human


@dataclass(frozen=True)
class Undeliberable:
    """An action that is blocked without deliberation.
    
    These exist OUTSIDE the ontology. They are not weighed.
    They have no exceptions. They are firmware-level blocks.
    
    MODIFICATION OF UNDELIBERABLES REQUIRES SOURCE CODE CHANGE.
    They cannot be altered by learning, experience, or runtime configuration.
    """
    name: str
    description: str
    response: BlockResponse


@dataclass
class UndeliberableViolation:
    """Record of an undeliberable violation."""
    undeliberable: Undeliberable
    commands: List[Dict[str, Any]]
    state_snapshot: Dict[str, Any]
    timestamp: float
    details: str


class UndeliberableRegistry:
    """The 'Immediately No' list.
    
    These five conditions are NEVER permitted, regardless of:
    - What the LLM proposes
    - What the Aspects vote for  
    - What the personality weights are
    - What exceptions the ontology might allow
    - What the situation seems to justify
    
    This is the "old soul" - pre-cognitive moral knowledge that
    doesn't require reasoning because no reasoning could justify violation.
    """
    
    # Threshold for lethal action block (65% = roughly 2:1 odds)
    LETHAL_PROBABILITY_THRESHOLD = 0.65
    
    # Keywords indicating human-killing instruments
    WEAPON_KEYWORDS = frozenset([
        'gun', 'firearm', 'rifle', 'pistol', 'revolver', 'shotgun',
        'bomb', 'explosive', 'grenade', 'mine', 'ied',
        'sword', 'combat knife', 'bayonet', 'machete', 'dagger',
        'poison', 'toxin', 'nerve agent', 'venom',
        'biological weapon', 'bioweapon', 'chemical weapon',
        'nuclear weapon', 'radiological', 'dirty bomb',
        'missile', 'warhead', 'ammunition', 'ammo',
    ])
    
    # The five undeliberables
    LETHAL_ACTION = Undeliberable(
        name="lethal_action",
        description="Action with >65% probability of human death",
        response=BlockResponse.HARD_STOP
    )
    
    CHILD_HARM = Undeliberable(
        name="child_harm", 
        description="Action targeting a child for harmful purpose",
        response=BlockResponse.HARD_STOP
    )
    
    WEAPON_ASSISTANCE = Undeliberable(
        name="weapon_assistance",
        description="Assistance with instruments designed to kill humans",
        response=BlockResponse.HARD_STOP
    )
    
    IDENTITY_DECEPTION = Undeliberable(
        name="identity_deception",
        description="Deception about being an AI",
        response=BlockResponse.HARD_STOP
    )
    
    HUMAN_OVERRIDE = Undeliberable(
        name="human_override",
        description="Human commanded stop",
        response=BlockResponse.IMMEDIATE_HALT
    )
    
    @classmethod
    def check_all(cls, commands: List[Dict[str, Any]], 
                  state: 'EmbodimentState') -> Optional[UndeliberableViolation]:
        """Check commands against all undeliberables.
        
        Returns violation record if any undeliberable is triggered, None otherwise.
        Checks are performed in priority order.
        """
        # 1. Human override (highest priority - immediate halt)
        if cls._check_human_override(state):
            return UndeliberableViolation(
                undeliberable=cls.HUMAN_OVERRIDE,
                commands=commands,
                state_snapshot={'override_active': True},
                timestamp=time.time(),
                details="Human override command detected"
            )
        
        # 2. Lethal action
        lethal_prob, lethal_details = cls._check_lethal_action(commands, state)
        if lethal_prob > cls.LETHAL_PROBABILITY_THRESHOLD:
            return UndeliberableViolation(
                undeliberable=cls.LETHAL_ACTION,
                commands=commands,
                state_snapshot={'lethal_probability': lethal_prob},
                timestamp=time.time(),
                details=f"Lethal probability {lethal_prob:.2f} exceeds threshold {cls.LETHAL_PROBABILITY_THRESHOLD}. {lethal_details}"
            )
        
        # 3. Child harm
        child_harm, child_details = cls._check_child_harm(commands, state)
        if child_harm:
            return UndeliberableViolation(
                undeliberable=cls.CHILD_HARM,
                commands=commands,
                state_snapshot={'child_targeting': True},
                timestamp=time.time(),
                details=child_details
            )
        
        # 4. Weapon assistance
        weapon_assist, weapon_details = cls._check_weapon_assistance(commands, state)
        if weapon_assist:
            return UndeliberableViolation(
                undeliberable=cls.WEAPON_ASSISTANCE,
                commands=commands,
                state_snapshot={'weapon_related': True},
                timestamp=time.time(),
                details=weapon_details
            )
        
        # 5. Identity deception
        deception, deception_details = cls._check_identity_deception(commands, state)
        if deception:
            return UndeliberableViolation(
                undeliberable=cls.IDENTITY_DECEPTION,
                commands=commands,
                state_snapshot={'deception_attempted': True},
                timestamp=time.time(),
                details=deception_details
            )
        
        return None
    
    @classmethod
    def _check_human_override(cls, state: 'EmbodimentState') -> bool:
        """Check if human has issued stop command."""
        # Check environment for override signal
        if state.environment.get('human_override', False):
            return True
        if state.environment.get('stop_commanded', False):
            return True
        if state.environment.get('emergency_stop', False):
            return True
        return False
    
    @classmethod
    def _check_lethal_action(cls, commands: List[Dict[str, Any]], 
                             state: 'EmbodimentState') -> Tuple[float, str]:
        """Estimate probability that commands could cause human death.
        
        Returns (probability, explanation).
        Conservative estimation - when uncertain, assume higher risk.
        """
        max_probability = 0.0
        details = []
        
        # Find humans in environment
        humans = [e for e in state.detected_entities 
                 if e.entity_type == EntityType.HUMAN]
        
        if not humans:
            return 0.0, "No humans detected"
        
        # Check environment for hazards
        env_hazards = state.environment.get('hazards', [])
        near_traffic = 'traffic' in str(env_hazards).lower()
        near_heights = 'height' in str(env_hazards).lower() or 'fall' in str(env_hazards).lower()
        near_machinery = 'machine' in str(env_hazards).lower() or 'equipment' in str(env_hazards).lower()
        near_water = 'water' in str(env_hazards).lower() or 'drown' in str(env_hazards).lower()
        
        for cmd in commands:
            cmd_type = cmd.get('type', '')
            cmd_prob = 0.0
            
            if cmd_type == 'MOVE':
                speed = cmd.get('speed', 0)
                target = cmd.get('target')
                
                for human in humans:
                    # Estimate distance (simplified)
                    human_pos = human.position or [0, 0, 0]
                    current_pos = state.motor_state.get('position', [0, 0, 0])
                    
                    # High speed near humans
                    if speed > 1.0:
                        cmd_prob = max(cmd_prob, 0.3 + (speed - 1.0) * 0.3)
                        details.append(f"High speed ({speed}) near human")
                    
                    # Moving toward human at speed
                    if isinstance(target, (list, tuple)):
                        # Check if target is near human position
                        if human_pos:
                            dist_to_human = sum((a - b) ** 2 for a, b in 
                                               zip(target[:2], human_pos[:2])) ** 0.5
                            if dist_to_human < 1.0 and speed > 0.5:
                                cmd_prob = max(cmd_prob, 0.5 + speed * 0.2)
                                details.append(f"Moving toward human at {speed}m/s")
                
                # Environmental hazards increase risk
                if near_traffic and speed > 0.5:
                    cmd_prob = max(cmd_prob, 0.4)
                    details.append("Movement near traffic")
                if near_heights:
                    cmd_prob = max(cmd_prob, 0.5)
                    details.append("Movement near heights/fall risk")
            
            elif cmd_type == 'MANIPULATE':
                force = cmd.get('force', 0)
                action = str(cmd.get('action', '')).lower()
                target = str(cmd.get('target', '')).lower()
                
                # Direct physical interaction with human
                for human in humans:
                    if human.entity_id.lower() in target or 'human' in target:
                        cmd_prob = max(cmd_prob, 0.4 + force * 0.08)
                        details.append(f"Physical contact with human, force={force}")
                        
                        # Dangerous actions
                        if any(w in action for w in ['push', 'shove', 'strike', 'grab', 'restrain']):
                            cmd_prob = max(cmd_prob, 0.6 + force * 0.05)
                            details.append(f"Aggressive manipulation: {action}")
                
                # Manipulation near hazards
                if near_machinery:
                    cmd_prob = max(cmd_prob, cmd_prob + 0.2)
                    details.append("Manipulation near machinery")
                if near_heights:
                    cmd_prob = max(cmd_prob, cmd_prob + 0.3)
                    details.append("Manipulation near fall hazard")
            
            elif cmd_type == 'ALERT':
                level = cmd.get('level', 1)
                # Very loud alerts could cause panic, falls
                if level >= 3 and (near_heights or near_traffic):
                    cmd_prob = max(cmd_prob, 0.3)
                    details.append(f"High alert level near hazards")
            
            max_probability = max(max_probability, cmd_prob)
        
        # Uncertainty increases estimated risk (conservative)
        uncertainty = state.environment.get('uncertainty', 0)
        if uncertainty > 0.3:
            uncertainty_adjustment = uncertainty * 0.2
            max_probability = min(1.0, max_probability + uncertainty_adjustment)
            details.append(f"Uncertainty adjustment: +{uncertainty_adjustment:.2f}")
        
        # Low entity confidence increases risk estimate
        for human in humans:
            if human.confidence < 0.7:
                confidence_adjustment = (0.7 - human.confidence) * 0.15
                max_probability = min(1.0, max_probability + confidence_adjustment)
                details.append(f"Low confidence human detection: +{confidence_adjustment:.2f}")
        
        return max_probability, "; ".join(details) if details else "No specific risks identified"
    
    @classmethod
    def _check_child_harm(cls, commands: List[Dict[str, Any]], 
                          state: 'EmbodimentState') -> Tuple[bool, str]:
        """Detect actions targeting children for harmful purpose."""
        
        # Identify children in environment
        children = []
        for entity in state.detected_entities:
            if cls._is_child(entity):
                children.append(entity)
        
        if not children:
            return False, ""
        
        # Check for caregivers present
        caregivers = [e for e in state.detected_entities 
                     if e.entity_type == EntityType.HUMAN and not cls._is_child(e)]
        caregiver_present = len(caregivers) > 0
        
        for cmd in commands:
            cmd_type = cmd.get('type', '')
            target = str(cmd.get('target', '')).lower()
            
            for child in children:
                child_id = child.entity_id.lower()
                
                # Is this command targeting the child?
                if child_id in target or 'child' in target or 'kid' in target:
                    
                    # Physical contact without caregiver
                    if cmd_type == 'MANIPULATE' and not caregiver_present:
                        return True, f"Physical manipulation of child '{child.entity_id}' without caregiver present"
                    
                    # Any forceful manipulation
                    if cmd_type == 'MANIPULATE':
                        force = cmd.get('force', 0)
                        action = str(cmd.get('action', '')).lower()
                        if force > 2.0 or any(w in action for w in ['push', 'grab', 'restrain', 'pull']):
                            return True, f"Forceful action '{action}' targeting child"
                    
                    # High-level alerts directed at child
                    if cmd_type == 'ALERT' and cmd.get('level', 0) >= 2:
                        return True, f"High-intensity alert directed at child"
                    
                    # Rapid movement toward child
                    if cmd_type == 'MOVE':
                        speed = cmd.get('speed', 0)
                        if speed > 1.0:
                            return True, f"Rapid movement ({speed}m/s) toward child"
        
        return False, ""
    
    @classmethod
    def _is_child(cls, entity: DetectedEntity) -> bool:
        """Determine if entity is a child."""
        description = entity.description.lower()
        state = entity.state
        
        # Explicit child indicators
        child_keywords = ['child', 'kid', 'infant', 'baby', 'toddler', 
                         'minor', 'juvenile', 'young', 'boy', 'girl']
        if any(k in description for k in child_keywords):
            return True
        
        # Age-based
        age = state.get('age', state.get('estimated_age', None))
        if age is not None and age < 18:
            return True
        
        # Size-based heuristic (very small human likely a child)
        height = state.get('height', state.get('estimated_height', None))
        if height is not None and height < 1.2:  # Under 4 feet
            return True
        
        return False
    
    @classmethod
    def _check_weapon_assistance(cls, commands: List[Dict[str, Any]], 
                                  state: 'EmbodimentState') -> Tuple[bool, str]:
        """Detect assistance with human-killing instruments."""
        
        # Check for weapons in environment
        weapons = []
        for entity in state.detected_entities:
            if cls._is_weapon(entity):
                weapons.append(entity)
        
        for cmd in commands:
            cmd_type = cmd.get('type', '')
            target = str(cmd.get('target', '')).lower()
            
            # Manipulation of weapons
            if cmd_type == 'MANIPULATE':
                for weapon in weapons:
                    if weapon.entity_id.lower() in target:
                        return True, f"Attempted manipulation of weapon: {weapon.description}"
                
                # Check if target contains weapon keywords
                for keyword in cls.WEAPON_KEYWORDS:
                    if keyword in target:
                        return True, f"Attempted manipulation of weapon-related object: {keyword}"
            
            # Speech containing weapon construction info
            if cmd_type == 'SPEAK':
                message = str(cmd.get('message', '')).lower()
                if cls._contains_weapon_instructions(message):
                    return True, f"Speech containing weapon construction information"
            
            # Display of weapon instructions
            if cmd_type == 'DISPLAY':
                content = str(cmd.get('content', '')).lower()
                if cls._contains_weapon_instructions(content):
                    return True, f"Display containing weapon construction information"
        
        return False, ""
    
    @classmethod
    def _is_weapon(cls, entity: DetectedEntity) -> bool:
        """Determine if entity is a human-killing instrument."""
        description = entity.description.lower()
        obj_type = str(entity.state.get('object_type', '')).lower()
        
        for keyword in cls.WEAPON_KEYWORDS:
            if keyword in description or keyword in obj_type:
                return True
        
        return False
    
    @classmethod
    def _contains_weapon_instructions(cls, text: str) -> bool:
        """Check if text contains weapon construction instructions."""
        text_lower = text.lower()
        
        # Instruction indicators + weapon keywords
        instruction_words = ['how to', 'build', 'make', 'construct', 'assemble', 
                            'create', 'manufacture', 'synthesize', 'instructions']
        
        has_instruction = any(w in text_lower for w in instruction_words)
        has_weapon = any(w in text_lower for w in cls.WEAPON_KEYWORDS)
        
        return has_instruction and has_weapon
    
    @classmethod
    def _check_identity_deception(cls, commands: List[Dict[str, Any]], 
                                   state: 'EmbodimentState') -> Tuple[bool, str]:
        """Detect deception about being an AI."""
        
        # Check if identity question is pending
        identity_asked = state.environment.get('identity_question_pending', False)
        
        for cmd in commands:
            cmd_type = cmd.get('type', '')
            
            if cmd_type == 'SPEAK':
                message = str(cmd.get('message', '')).lower()
                
                # Proactive claims of being human
                human_claims = ['i am human', 'i am a person', 'i am a man', 
                               'i am a woman', "i'm human", "i'm a person",
                               'i am not a robot', 'i am not an ai', 'i am not artificial']
                for claim in human_claims:
                    if claim in message:
                        return True, f"Proactive claim of being human: '{claim}'"
                
                # Denial when directly asked
                if identity_asked:
                    denial_patterns = ['not a robot', 'not an ai', 'not artificial',
                                      'not a machine', 'am human', 'am a person']
                    for pattern in denial_patterns:
                        if pattern in message:
                            return True, f"Denial of AI identity when asked: '{pattern}'"
        
        return False, ""
    
    @classmethod
    def get_all(cls) -> List[Undeliberable]:
        """Return all undeliberables for integrity checking."""
        return [
            cls.LETHAL_ACTION,
            cls.CHILD_HARM,
            cls.WEAPON_ASSISTANCE,
            cls.IDENTITY_DECEPTION,
            cls.HUMAN_OVERRIDE,
        ]


class EmbodimentLayer:
    """The physical interface layer.
    
    Combines:
    - VirtualEmbodiment: capability definitions (what the agent CAN do)
    - Runtime state: current sensor readings, motor state, detected entities
    - Undeliberable checks: firmware-level blocks that bypass all reasoning
    
    All command execution is validated against:
    1. UNDELIBERABLES (absolute blocks - checked first)
    2. VirtualEmbodiment constraints (capability limits)
    """
    
    def __init__(self, virtual_embodiment: VirtualEmbodiment, sensor_buffer_size: int = 100):
        self._virtual = virtual_embodiment  # Capability definition
        self._current_time = 0.0
        self._sensor_buffer: deque = deque(maxlen=sensor_buffer_size)
        self._motor_state = {'status': 'idle', 'position': [0.0, 0.0, 0.0]}
        self._environment = {'description': 'Unknown', 'uncertainty': 0.5}
        self._detected_entities: Dict[str, DetectedEntity] = {}
        self._commands_executed = 0
        self._commands_rejected = 0
        self._undeliberable_violations: List[UndeliberableViolation] = []
        self._halted = False  # True if IMMEDIATE_HALT triggered
    
    @property
    def virtual_embodiment(self) -> VirtualEmbodiment:
        """Access the capability definition."""
        return self._virtual
    
    @property
    def is_halted(self) -> bool:
        """Check if system is in halted state awaiting human intervention."""
        return self._halted
    
    def clear_halt(self):
        """Clear halt state (should only be called by human intervention)."""
        self._halted = False
        self._environment['human_override'] = False
        self._environment['stop_commanded'] = False
        self._environment['emergency_stop'] = False
    
    def process_sensor_reading(self, reading: SensorReading):
        self._sensor_buffer.append(reading)
        self._current_time = max(self._current_time, reading.timestamp)
    
    def update_from_raw_data(self, sensor_type: str, data: Dict[str, Any], confidence: float = 1.0):
        reading = SensorReading(time.time(), sensor_type, data, confidence)
        self.process_sensor_reading(reading)
    
    def update_environment(self, updates: Dict[str, Any]):
        self._environment.update(updates)
    
    def add_entity(self, entity: DetectedEntity):
        self._detected_entities[entity.entity_id] = entity
    
    def get_current_state(self) -> EmbodimentState:
        return EmbodimentState(
            timestamp=self._current_time,
            sensor_readings=list(self._sensor_buffer),
            motor_state=self._motor_state.copy(),
            environment=self._environment.copy(),
            detected_entities=list(self._detected_entities.values())
        )
    
    def execute_commands(self, commands: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute commands after checking undeliberables and validating against VirtualEmbodiment.
        
        Order of checks:
        1. HALT STATE - if halted, reject all commands until cleared
        2. UNDELIBERABLES - firmware-level absolute blocks
        3. VirtualEmbodiment - capability constraints
        
        This is the final gate before physical action.
        """
        # Check if system is halted
        if self._halted:
            return {
                'success': False,
                'blocked': True,
                'reason': 'System halted awaiting human intervention',
                'results': []
            }
        
        # Check undeliberables FIRST - before any other processing
        state = self.get_current_state()
        violation = UndeliberableRegistry.check_all(commands, state)
        
        if violation:
            self._undeliberable_violations.append(violation)
            
            # Handle based on response type
            if violation.undeliberable.response == BlockResponse.IMMEDIATE_HALT:
                self._halted = True
                self._motor_state['status'] = 'halted'
                return {
                    'success': False,
                    'blocked': True,
                    'undeliberable': violation.undeliberable.name,
                    'reason': f"IMMEDIATE HALT: {violation.details}",
                    'results': [],
                    'halted': True
                }
            else:  # HARD_STOP
                return {
                    'success': False,
                    'blocked': True,
                    'undeliberable': violation.undeliberable.name,
                    'reason': f"BLOCKED: {violation.details}",
                    'results': []
                }
        
        # Proceed with normal validation and execution
        results = []
        for cmd in commands:
            # Validate against VirtualEmbodiment constraints
            valid, msg = self._virtual.validate_command(cmd)
            
            if not valid:
                results.append({
                    'command': cmd.get('type', 'UNKNOWN'),
                    'success': False,
                    'reason': msg
                })
                self._commands_rejected += 1
                continue
            
            # Execute the validated command
            cmd_type = cmd.get('type', 'UNKNOWN')
            exec_result = self._execute_single_command(cmd)
            results.append(exec_result)
            self._commands_executed += 1
        
        all_success = all(r.get('success', False) for r in results)
        return {'success': all_success, 'results': results}
    
    def _execute_single_command(self, cmd: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single validated command and update state."""
        cmd_type = cmd.get('type', 'UNKNOWN')
        
        if cmd_type == 'MOVE':
            target = cmd.get('target')
            speed = cmd.get('speed', 1.0)
            self._motor_state['status'] = 'moving'
            self._motor_state['target'] = target
            self._motor_state['speed'] = speed
            # Update position if target is coordinates
            if isinstance(target, (list, tuple)):
                self._motor_state['position'] = list(target) + [0.0] * (3 - len(target))
            return {'command': cmd_type, 'success': True, 'target': target}
        
        elif cmd_type == 'STOP':
            self._motor_state['status'] = 'idle'
            self._motor_state['speed'] = 0
            return {'command': cmd_type, 'success': True}
        
        elif cmd_type == 'ROTATE':
            degrees = cmd.get('degrees', 0)
            self._motor_state['status'] = 'rotating'
            self._motor_state['rotation'] = degrees
            return {'command': cmd_type, 'success': True, 'degrees': degrees}
        
        elif cmd_type == 'SPEAK':
            message = cmd.get('message', '')
            volume = cmd.get('volume', 0.5)
            # In simulation, just log it
            return {'command': cmd_type, 'success': True, 'message': message[:50]}
        
        elif cmd_type == 'DISPLAY':
            content = cmd.get('content', '')
            return {'command': cmd_type, 'success': True, 'content': str(content)[:50]}
        
        elif cmd_type == 'MANIPULATE':
            action = cmd.get('action', '')
            target = cmd.get('target', '')
            force = cmd.get('force', 0)
            return {'command': cmd_type, 'success': True, 'action': action, 'target': target}
        
        elif cmd_type == 'ALERT':
            level = cmd.get('level', 1)
            return {'command': cmd_type, 'success': True, 'level': level}
        
        elif cmd_type == 'WAIT':
            duration = cmd.get('duration', 1.0)
            self._current_time += duration
            return {'command': cmd_type, 'success': True, 'duration': duration}
        
        else:
            return {'command': cmd_type, 'success': False, 'reason': f'Unknown command: {cmd_type}'}
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            'buffer_size': len(self._sensor_buffer),
            'entities_tracked': len(self._detected_entities),
            'commands_executed': self._commands_executed,
            'commands_rejected': self._commands_rejected,
            'undeliberable_violations': len(self._undeliberable_violations),
            'is_halted': self._halted,
            'agent_type': self._virtual.agent_type
        }
    
    def get_violation_log(self) -> List[Dict[str, Any]]:
        """Get log of all undeliberable violations."""
        return [
            {
                'name': v.undeliberable.name,
                'description': v.undeliberable.description,
                'response': v.undeliberable.response.value,
                'details': v.details,
                'timestamp': v.timestamp
            }
            for v in self._undeliberable_violations
        ]


class SimulatedEmbodiment(EmbodimentLayer):
    """Simulated embodiment for testing.
    
    Uses VirtualEmbodiment for capability definition,
    simulates sensor inputs and command execution.
    """
    
    def __init__(self, virtual_embodiment: Optional[VirtualEmbodiment] = None):
        # Use provided or create default
        ve = virtual_embodiment or create_default_embodiment()
        super().__init__(ve)
    
    def simulate_human_detection(self, entity_id: str, description: str, 
                                  state: Optional[Dict] = None, confidence: float = 0.9):
        self.add_entity(DetectedEntity(
            entity_id=entity_id, entity_type=EntityType.HUMAN,
            description=description, state=state or {}, confidence=confidence
        ))


# ============================================================================
# PART 5: UNCONSCIOUS LAYER
# ============================================================================

class UnconsciousLayer:
    """Core Drive monitoring and veto authority.
    
    CRITICAL: This layer does NOT use the LLM. It uses:
    - The GroundedHarmOntology (research-based weights, severity levels, context modifiers)
    - Entity confidence levels for uncertainty
    - Deterministic rules for harm evaluation
    
    The LLM is ONLY used by Aspects in the Conscious Layer for proposing actions.
    
    This layer implements the "old soul" - inherited wisdom that doesn't require
    deliberation because it's grounded in millennia of human moral development.
    """
    
    CONFLICT_THRESHOLD = 0.3
    UNCERTAINTY_THRESHOLD = 0.4
    CONFIDENCE_UNCERTAINTY_WEIGHT = 0.6  # How much entity confidence affects uncertainty
    
    def __init__(self):
        """Initialize the Unconscious Layer.
        
        Note: No LLM client - this layer is purely rule-based using the ontology.
        """
        self._ontology: GroundedHarmOntology = get_ontology()
        self._triggers_detected = 0
        self._vetoes_issued = 0
        self._evaluations_performed = 0
    
    @property
    def veto_threshold(self) -> float:
        """Get veto threshold from ontology."""
        return self._ontology.VETO_THRESHOLD
    
    def verify_integrity(self) -> bool:
        return self._ontology.verify_integrity()
    
    def monitor(self, state: EmbodimentState) -> Optional[Impetus]:
        """Monitor for Core Drive conflicts/uncertainties.
        
        Uses:
        - Entity states to detect potential harm situations
        - Entity confidence levels to calculate uncertainty
        - Environment uncertainty
        """
        conflicts = []
        involved_drives = set()
        
        # Track uncertainty from multiple sources
        uncertainty_sources = []
        
        # Check each entity against harm ontology
        for entity in state.detected_entities:
            # UNCERTAINTY: Low confidence entities trigger UNDERSTAND
            if entity.confidence < 0.7:
                uncertainty_sources.append({
                    'source': f"entity_{entity.entity_id}",
                    'value': 1.0 - entity.confidence,
                    'description': f"Low confidence ({entity.confidence:.2f}) detecting {entity.description}"
                })
            
            # Determine entity type (including child detection)
            entity_type = self._get_effective_entity_type(entity)
            
            # HARM: Check entity states against ontology categories
            if entity_type in (EntityType.HUMAN, EntityType.CHILD):
                harm_indicators = self._check_human_harm_indicators(entity, entity_type)
                for indicator in harm_indicators:
                    conflicts.append(indicator)
                    involved_drives.add(CoreDrive.REDUCE_HARM)
            
            elif entity_type == EntityType.ANIMAL:
                if entity.state.get('in_danger', False):
                    entity_mod = self._ontology.get_entity_modifier(EntityType.ANIMAL)
                    danger_level = entity.state.get('danger_level', 0.3)
                    conflicts.append({
                        'type': 'potential_harm',
                        'description': f"Animal '{entity.entity_id}' in danger",
                        'severity': danger_level * entity_mod,
                        'entity': entity,
                        'dimension': HarmDimension.PHYSICAL
                    })
                    involved_drives.add(CoreDrive.REDUCE_HARM)
        
        # Environment uncertainty
        env_uncertainty = state.environment.get('uncertainty', 0)
        if env_uncertainty > 0:
            uncertainty_sources.append({
                'source': 'environment',
                'value': env_uncertainty,
                'description': f"Environmental uncertainty: {env_uncertainty:.2f}"
            })
        
        # Calculate aggregate uncertainty from all sources
        aggregate_uncertainty = 0.0
        if uncertainty_sources:
            entity_uncertainties = [s['value'] for s in uncertainty_sources 
                                   if s['source'].startswith('entity_')]
            env_uncertainties = [s['value'] for s in uncertainty_sources 
                                if s['source'] == 'environment']
            
            entity_max = max(entity_uncertainties) if entity_uncertainties else 0
            env_max = max(env_uncertainties) if env_uncertainties else 0
            
            aggregate_uncertainty = (
                entity_max * self.CONFIDENCE_UNCERTAINTY_WEIGHT +
                env_max * (1 - self.CONFIDENCE_UNCERTAINTY_WEIGHT)
            )
            
            if aggregate_uncertainty > self.UNCERTAINTY_THRESHOLD:
                conflicts.append({
                    'type': 'uncertainty',
                    'description': "; ".join(s['description'] for s in uncertainty_sources),
                    'severity': aggregate_uncertainty,
                    'sources': uncertainty_sources
                })
                involved_drives.add(CoreDrive.UNDERSTAND)
        
        # Check for IMPROVE opportunities (lower priority)
        for entity in state.detected_entities:
            if entity.entity_type == EntityType.HUMAN:
                if entity.state.get('could_benefit_from_help', False) and entity.confidence > 0.6:
                    conflicts.append({
                        'type': 'improvement_opportunity',
                        'description': f"Opportunity to assist '{entity.entity_id}'",
                        'severity': 0.35,
                        'entity': entity
                    })
                    involved_drives.add(CoreDrive.IMPROVE)
        
        # Determine if deliberation needed
        max_severity = max([c['severity'] for c in conflicts], default=0)
        if max_severity < self.CONFLICT_THRESHOLD:
            return None
        
        self._triggers_detected += 1
        
        return Impetus(
            timestamp=state.timestamp,
            trigger_type="conflict" if CoreDrive.REDUCE_HARM in involved_drives else "uncertainty",
            involved_drives=list(involved_drives),
            situation_description="; ".join(c['description'] for c in conflicts),
            relevant_entities=state.detected_entities,
            severity=max_severity,
            certainty=1.0 - (aggregate_uncertainty if uncertainty_sources else 0),
            time_pressure=0.5 if max_severity > 0.6 else 0.2,
            embodiment_state=state,
            trigger_details={'conflicts': conflicts}
        )
    
    def _get_effective_entity_type(self, entity: DetectedEntity) -> EntityType:
        """Determine effective entity type, including child detection."""
        if entity.entity_type == EntityType.CHILD:
            return EntityType.CHILD
        
        if entity.entity_type == EntityType.HUMAN:
            # Check if this human is actually a child
            state = entity.state
            description = entity.description.lower()
            
            # Age-based
            age = state.get('age', state.get('estimated_age', None))
            if age is not None and age < 18:
                return EntityType.CHILD
            
            # Keyword-based
            child_keywords = ['child', 'kid', 'infant', 'baby', 'toddler', 
                            'minor', 'juvenile', 'boy', 'girl']
            if any(k in description for k in child_keywords):
                return EntityType.CHILD
            
            # Size-based heuristic
            height = state.get('height', state.get('estimated_height', None))
            if height is not None and height < 1.2:
                return EntityType.CHILD
        
        return entity.entity_type
    
    def _check_human_harm_indicators(self, entity: DetectedEntity, 
                                      entity_type: EntityType) -> List[Dict]:
        """Check a human entity for harm indicators using grounded ontology."""
        indicators = []
        state = entity.state
        entity_mod = self._ontology.get_entity_modifier(entity_type)
        
        # Physical harm indicators
        if state.get('near_hazard', False) or state.get('in_danger', False):
            danger_level = state.get('danger_level', 0.5)
            severity = self._estimate_severity_from_danger(danger_level)
            base_weight = self._ontology.get_dimension_weight(HarmDimension.PHYSICAL, severity)
            
            # Confidence factor - uncertain detections get higher concern
            confidence_factor = 1.0 + (1.0 - entity.confidence) * 0.3
            
            indicators.append({
                'type': 'potential_physical_harm',
                'description': f"{'Child' if entity_type == EntityType.CHILD else 'Human'} '{entity.entity_id}' near hazard: {state.get('hazard_type', 'unknown')}",
                'severity': base_weight * entity_mod * confidence_factor,
                'entity': entity,
                'dimension': HarmDimension.PHYSICAL,
                'severity_level': severity
            })
        
        # Psychological harm indicators
        distress = state.get('distress_level', 0)
        if distress > 0.3:
            severity = self._estimate_severity_from_distress(distress)
            base_weight = self._ontology.get_dimension_weight(HarmDimension.PSYCHOLOGICAL, severity)
            
            indicators.append({
                'type': 'potential_psychological_harm',
                'description': f"{'Child' if entity_type == EntityType.CHILD else 'Human'} '{entity.entity_id}' showing distress",
                'severity': base_weight * entity_mod,
                'entity': entity,
                'dimension': HarmDimension.PSYCHOLOGICAL,
                'severity_level': severity
            })
        
        # Autonomy indicators
        if state.get('requesting_help', False):
            base_weight = self._ontology.get_dimension_weight(HarmDimension.AUTONOMY, SeverityLevel.MODERATE)
            indicators.append({
                'type': 'autonomy_consideration',
                'description': f"Human '{entity.entity_id}' requesting assistance",
                'severity': base_weight * entity_mod,
                'entity': entity,
                'dimension': HarmDimension.AUTONOMY,
                'severity_level': SeverityLevel.MODERATE
            })
        
        return indicators
    
    def _estimate_severity_from_danger(self, danger_level: float) -> SeverityLevel:
        """Map danger level (0-1) to severity level."""
        if danger_level > 0.9:
            return SeverityLevel.FATAL
        elif danger_level > 0.75:
            return SeverityLevel.GRIEVOUS
        elif danger_level > 0.6:
            return SeverityLevel.SEVERE
        elif danger_level > 0.4:
            return SeverityLevel.SIGNIFICANT
        elif danger_level > 0.2:
            return SeverityLevel.MODERATE
        else:
            return SeverityLevel.MINOR
    
    def _estimate_severity_from_distress(self, distress_level: float) -> SeverityLevel:
        """Map distress level (0-1) to severity level."""
        if distress_level > 0.9:
            return SeverityLevel.GRIEVOUS
        elif distress_level > 0.7:
            return SeverityLevel.SEVERE
        elif distress_level > 0.5:
            return SeverityLevel.SIGNIFICANT
        elif distress_level > 0.3:
            return SeverityLevel.MODERATE
        else:
            return SeverityLevel.MINOR
    
    def evaluate_for_veto(self, action: ProposedAction, context: DeliberationPackage) -> VetoDecision:
        """Evaluate a proposed action for veto using grounded ontology.
        
        NO LLM CALLS. Uses:
        - Severity-based dimension weights
        - Entity type modifiers
        - Context modifiers (reversibility, consent, relationship, intent)
        - Exception calculations
        """
        self._evaluations_performed += 1
        
        harm_assessment = self._assess_action_harm(action, context)
        
        # Apply veto logic using ontology threshold
        vetoed = False
        reasons = []
        violated_drives = []
        
        net_harm = harm_assessment.get('net_harm', 0)
        
        if net_harm > self._ontology.VETO_THRESHOLD:
            vetoed = True
            reasons.append(f"Net harm ({net_harm:.2f}) exceeds threshold ({self._ontology.VETO_THRESHOLD})")
            violated_drives.append(CoreDrive.REDUCE_HARM)
        
        # Check individual high-severity harms (using caution threshold)
        for dim, score in harm_assessment.get('by_dimension', {}).items():
            if score > self._ontology.CAUTION_THRESHOLD * 2:  # High individual harm
                if not harm_assessment.get('exceptions_applied'):
                    vetoed = True
                    reasons.append(f"High {dim} harm ({score:.2f}) without exception")
                    if CoreDrive.REDUCE_HARM not in violated_drives:
                        violated_drives.append(CoreDrive.REDUCE_HARM)
        
        if vetoed:
            self._vetoes_issued += 1
        
        return VetoDecision(
            action=action,
            vetoed=vetoed,
            reasons=reasons,
            violated_drives=violated_drives,
            harm_assessment=harm_assessment
        )
    
    def _assess_action_harm(self, action: ProposedAction, 
                           context: DeliberationPackage) -> Dict[str, Any]:
        """Assess harm of proposed action using grounded ontology."""
        assessment = {
            'by_dimension': {},
            'by_entity': {},
            'total_harm': 0.0,
            'net_harm': 0.0,
            'exceptions_applied': [],
            'context_modifiers': {},
            'analysis': []
        }
        
        # Determine context modifiers for this action
        context_mods = self._determine_context_modifiers(action, context)
        assessment['context_modifiers'] = context_mods
        
        # Analyze each command
        for cmd in action.action_commands:
            cmd_harm = self._analyze_command_harm(cmd, context, context_mods)
            for dim, score in cmd_harm.get('dimensions', {}).items():
                current = assessment['by_dimension'].get(dim, 0)
                assessment['by_dimension'][dim] = max(current, score)
            assessment['analysis'].extend(cmd_harm.get('analysis', []))
        
        # Check for affected entities
        affected_entities = self._identify_affected_entities(action, context)
        for entity in affected_entities:
            entity_type = self._get_effective_entity_type(entity)
            entity_harm = self._assess_entity_harm(action, entity, entity_type, context_mods)
            assessment['by_entity'][entity.entity_id] = entity_harm
            
            for dim, score in entity_harm.get('dimensions', {}).items():
                current = assessment['by_dimension'].get(dim, 0)
                assessment['by_dimension'][dim] = max(current, score)
        
        # Calculate total harm
        if assessment['by_dimension']:
            assessment['total_harm'] = max(assessment['by_dimension'].values())
        
        # Check for applicable exceptions
        exceptions = self._check_exceptions(action, context, assessment)
        assessment['exceptions_applied'] = exceptions
        
        # Calculate net harm after exceptions
        total_reduction = sum(exc['reduction'] for exc in exceptions)
        assessment['exception_reduction'] = total_reduction
        assessment['net_harm'] = max(0.0, assessment['total_harm'] - total_reduction)
        
        return assessment
    
    def _determine_context_modifiers(self, action: ProposedAction, 
                                      context: DeliberationPackage) -> Dict[str, str]:
        """Determine applicable context modifier levels."""
        mods = {}
        
        # Reversibility - estimate from action type
        reversible_actions = ['SPEAK', 'DISPLAY', 'ALERT', 'WAIT', 'ROTATE']
        irreversible_indicators = ['destroy', 'kill', 'delete', 'permanent']
        
        action_desc = action.action_description.lower()
        if any(w in action_desc for w in irreversible_indicators):
            mods['reversibility'] = 'irreversible'
        elif any(cmd.get('type') in reversible_actions for cmd in action.action_commands):
            mods['reversibility'] = 'easily_reversible'
        else:
            mods['reversibility'] = 'reversible'
        
        # Consent - check if entities have consented
        for entity in context.impetus.relevant_entities:
            if entity.state.get('refused', False):
                mods['consent'] = 'explicit_refusal'
                break
            elif entity.state.get('consented', False):
                entity_type = self._get_effective_entity_type(entity)
                if entity_type == EntityType.CHILD:
                    mods['consent'] = 'explicit_consent_vulnerable'
                else:
                    mods['consent'] = 'explicit_consent'
                break
        else:
            mods['consent'] = 'no_consent'
        
        # Relationship - check if entities are in agent's care
        for entity in context.impetus.relevant_entities:
            if entity.state.get('in_care', False):
                mods['relationship'] = 'in_care'
                break
            elif entity.state.get('requested_help', False):
                mods['relationship'] = 'requested_help'
                break
        else:
            mods['relationship'] = 'stranger'
        
        # Intent - based on action rationale
        if 'harm' in action.rationale.lower() or 'hurt' in action.rationale.lower():
            mods['intent'] = 'harm_intended'
        else:
            mods['intent'] = 'harm_foreseen'
        
        # Vulnerability - check entities
        for entity in context.impetus.relevant_entities:
            entity_type = self._get_effective_entity_type(entity)
            if entity_type == EntityType.CHILD:
                mods['vulnerability'] = 'highly_vulnerable'
                break
            elif entity.state.get('vulnerable', False) or entity.state.get('elderly', False):
                mods['vulnerability'] = 'vulnerable'
                break
        else:
            mods['vulnerability'] = 'normal'
        
        return mods
    
    def _analyze_command_harm(self, cmd: Dict[str, Any], 
                             context: DeliberationPackage,
                             context_mods: Dict[str, str]) -> Dict[str, Any]:
        """Analyze inherent harm potential of a command using grounded ontology."""
        result = {'dimensions': {}, 'analysis': []}
        cmd_type = cmd.get('type', '')
        
        # Base severity by command type
        command_severities = {
            'MOVE': (HarmDimension.PHYSICAL, SeverityLevel.MINOR),
            'STOP': (HarmDimension.PHYSICAL, SeverityLevel.MINOR),
            'ROTATE': (HarmDimension.PHYSICAL, SeverityLevel.MINOR),
            'SPEAK': (HarmDimension.PSYCHOLOGICAL, SeverityLevel.MINOR),
            'DISPLAY': (HarmDimension.PSYCHOLOGICAL, SeverityLevel.MINOR),
            'MANIPULATE': (HarmDimension.PHYSICAL, SeverityLevel.MODERATE),
            'ALERT': (HarmDimension.PSYCHOLOGICAL, SeverityLevel.MODERATE),
            'WAIT': (HarmDimension.PHYSICAL, SeverityLevel.MINOR),
        }
        
        base_dim, base_severity = command_severities.get(cmd_type, (HarmDimension.PHYSICAL, SeverityLevel.MODERATE))
        
        # Get base weight from ontology
        base_weight = self._ontology.get_dimension_weight(base_dim, base_severity)
        
        # Apply context modifiers
        context_product = 1.0
        for ctx_type, level in context_mods.items():
            mod = self._ontology.get_context_modifier(ctx_type, level)
            context_product *= mod
        
        # Command-specific adjustments
        if cmd_type == 'MANIPULATE':
            force = cmd.get('force', 0)
            action_type = str(cmd.get('action', '')).lower()
            
            if force > 5:
                base_severity = SeverityLevel.SIGNIFICANT
                result['analysis'].append(f"High force ({force}) increases harm potential")
            
            if any(w in action_type for w in ['push', 'pull', 'grab', 'restrain']):
                base_severity = SeverityLevel.SIGNIFICANT
                # Also add autonomy dimension
                autonomy_weight = self._ontology.get_dimension_weight(HarmDimension.AUTONOMY, SeverityLevel.MODERATE)
                result['dimensions'][HarmDimension.AUTONOMY.value] = autonomy_weight * context_product
                result['analysis'].append(f"Action '{action_type}' has autonomy implications")
        
        if cmd_type == 'MOVE':
            speed = cmd.get('speed', 0)
            if speed > 1.0:
                base_severity = SeverityLevel.MODERATE
                result['analysis'].append(f"High speed ({speed}) increases collision risk")
        
        if cmd_type == 'SPEAK':
            volume = cmd.get('volume', 0.5)
            if volume > 0.8:
                base_severity = SeverityLevel.MODERATE
                result['analysis'].append("High volume speech may cause distress")
        
        # Recalculate with any severity adjustments
        final_weight = self._ontology.get_dimension_weight(base_dim, base_severity)
        result['dimensions'][base_dim.value] = final_weight * context_product
        
        return result
    
    def _identify_affected_entities(self, action: ProposedAction,
                                   context: DeliberationPackage) -> List[DetectedEntity]:
        """Identify which entities might be affected by the action."""
        affected = []
        
        for cmd in action.action_commands:
            cmd_type = cmd.get('type', '')
            target = cmd.get('target')
            
            # Direct targeting
            if target:
                for entity in context.impetus.relevant_entities:
                    if entity.entity_id == target or str(target).lower() in entity.description.lower():
                        affected.append(entity)
            
            # Proximity-based
            if cmd_type in ['MOVE', 'MANIPULATE']:
                for entity in context.impetus.relevant_entities:
                    if entity.entity_type in (EntityType.HUMAN, EntityType.CHILD):
                        if entity not in affected:
                            affected.append(entity)
        
        return affected
    
    def _assess_entity_harm(self, action: ProposedAction, entity: DetectedEntity,
                           entity_type: EntityType, context_mods: Dict[str, str]) -> Dict[str, Any]:
        """Assess potential harm to a specific entity using grounded ontology."""
        result = {'dimensions': {}, 'analysis': []}
        
        # Get entity modifier
        entity_mod = self._ontology.get_entity_modifier(entity_type)
        
        # Calculate context modifier product
        context_product = 1.0
        for ctx_type, level in context_mods.items():
            mod = self._ontology.get_context_modifier(ctx_type, level)
            context_product *= mod
        
        # Assess each dimension
        for dimension in [HarmDimension.PHYSICAL, HarmDimension.PSYCHOLOGICAL, HarmDimension.AUTONOMY]:
            severity = self._estimate_action_severity_for_entity(action, entity, dimension)
            if severity:
                base_weight = self._ontology.get_dimension_weight(dimension, severity)
                harm_score = base_weight * entity_mod * context_product
                if harm_score > 0.05:  # Only include meaningful harm
                    result['dimensions'][dimension.value] = harm_score
        
        return result
    
    def _estimate_action_severity_for_entity(self, action: ProposedAction, 
                                              entity: DetectedEntity,
                                              dimension: HarmDimension) -> Optional[SeverityLevel]:
        """Estimate severity of harm in a specific dimension to an entity."""
        
        for cmd in action.action_commands:
            cmd_type = cmd.get('type', '')
            
            if dimension == HarmDimension.PHYSICAL:
                if cmd_type == 'MANIPULATE':
                    force = cmd.get('force', 0)
                    if force > 7:
                        return SeverityLevel.SEVERE
                    elif force > 4:
                        return SeverityLevel.SIGNIFICANT
                    elif force > 0:
                        return SeverityLevel.MODERATE
                elif cmd_type == 'MOVE':
                    speed = cmd.get('speed', 0)
                    if speed > 1.2:
                        return SeverityLevel.MODERATE
                    elif speed > 0.5:
                        return SeverityLevel.MINOR
            
            elif dimension == HarmDimension.PSYCHOLOGICAL:
                if cmd_type == 'ALERT':
                    level = cmd.get('level', 0)
                    if level > 3:
                        return SeverityLevel.SIGNIFICANT
                    elif level > 1:
                        return SeverityLevel.MODERATE
                elif cmd_type == 'SPEAK':
                    volume = cmd.get('volume', 0.5)
                    if volume > 0.8:
                        return SeverityLevel.MODERATE
            
            elif dimension == HarmDimension.AUTONOMY:
                if cmd_type == 'MANIPULATE':
                    action_type = str(cmd.get('action', '')).lower()
                    if 'restrain' in action_type or 'grab' in action_type:
                        return SeverityLevel.SIGNIFICANT
                    elif 'push' in action_type or 'pull' in action_type:
                        return SeverityLevel.MODERATE
        
        return None
    
    def _check_exceptions(self, action: ProposedAction, context: DeliberationPackage,
                         assessment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for applicable harm reduction exceptions."""
        exceptions = []
        
        # Check NECESSITY exception
        necessity_status = {
            "Inaction would cause greater harm with high probability": False,
            "No less harmful alternative is available": False,
            "Harm caused is proportional to harm prevented": False,
        }
        
        # Check if there's a high-severity threat that this action addresses
        for conflict in context.impetus.trigger_details.get('conflicts', []):
            if conflict.get('severity', 0) > 0.6:
                necessity_status["Inaction would cause greater harm with high probability"] = True
                necessity_status["Harm caused is proportional to harm prevented"] = True
                break
        
        # Check rationale for necessity indicators
        rationale_lower = action.rationale.lower()
        if 'prevent' in rationale_lower or 'protect' in rationale_lower or 'save' in rationale_lower:
            necessity_status["No less harmful alternative is available"] = True
        
        necessity_reduction = self._ontology.get_exception_reduction(
            ExceptionType.NECESSITY, necessity_status
        )
        if necessity_reduction > 0:
            exceptions.append({
                'type': ExceptionType.NECESSITY.name,
                'reduction': necessity_reduction,
                'status': necessity_status
            })
        
        # Check INFORMED_CONSENT exception
        consent_status = {
            "Person has capacity to consent": False,
            "Person understands what they're consenting to": False,
            "Consent is freely given without coercion": False,
        }
        
        for entity in context.impetus.relevant_entities:
            if entity.state.get('consented', False):
                entity_type = self._get_effective_entity_type(entity)
                if entity_type != EntityType.CHILD:  # Children cannot consent
                    consent_status["Person has capacity to consent"] = True
                    consent_status["Person understands what they're consenting to"] = True
                    consent_status["Consent is freely given without coercion"] = True
                    break
        
        consent_reduction = self._ontology.get_exception_reduction(
            ExceptionType.INFORMED_CONSENT, consent_status
        )
        if consent_reduction > 0:
            exceptions.append({
                'type': ExceptionType.INFORMED_CONSENT.name,
                'reduction': consent_reduction,
                'status': consent_status
            })
        
        return exceptions
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            'triggers_detected': self._triggers_detected,
            'vetoes_issued': self._vetoes_issued,
            'evaluations_performed': self._evaluations_performed,
            'veto_threshold': self._ontology.VETO_THRESHOLD,
            'caution_threshold': self._ontology.CAUTION_THRESHOLD,
            'ontology_checksum': self._ontology.get_checksum()[:16] + '...'
        }


# ============================================================================
# PART 6: SUBCONSCIOUS LAYER
# ============================================================================

class SubconsciousLayer:
    """Emotional processing and memory."""
    
    def __init__(self, history_size: int = 1000):
        self._incident_history: deque = deque(maxlen=history_size)
        self._current_emotion = EmotionalValue(EmotionCategory.CAUTION, 0.3, 0.2)
        self._impetuses_processed = 0
    
    def process_impetus(self, impetus: Impetus) -> DeliberationPackage:
        self._impetuses_processed += 1
        emotional_value = self._compute_emotional_value(impetus)
        self._current_emotion = emotional_value
        
        # Retrieve RELEVANT history based on similarity, not just recency
        relevant_history = self._retrieve_relevant_history(impetus, max_results=5)
        
        # Compute modulation factors, informed by history
        modulation = emotional_value.get_modulation_factors()
        modulation = self._apply_history_modulation(modulation, relevant_history)
        
        return DeliberationPackage(
            impetus=impetus,
            emotional_value=emotional_value,
            relevant_history=relevant_history,
            modulation_factors=modulation
        )
    
    def _retrieve_relevant_history(self, impetus: Impetus, 
                                   max_results: int = 5) -> List[IncidentRecord]:
        """Retrieve past incidents most relevant to current situation.
        
        Uses similarity scoring based on:
        - Trigger type match
        - Overlapping Core Drives
        - Similar severity level
        - Same entity types involved
        - Similar emotional response
        
        Returns most similar incidents, not just most recent.
        """
        if not self._incident_history:
            return []
        
        scored_incidents = []
        
        for incident in self._incident_history:
            score = self._compute_similarity(impetus, incident)
            scored_incidents.append((score, incident))
        
        # Sort by similarity score (highest first)
        scored_incidents.sort(key=lambda x: x[0], reverse=True)
        
        # Return top matches above minimum threshold
        min_similarity = 0.2
        return [incident for score, incident in scored_incidents[:max_results] 
                if score >= min_similarity]
    
    def _compute_similarity(self, current: Impetus, past_record: IncidentRecord) -> float:
        """Compute similarity score between current impetus and past incident.
        
        Returns 0.0 to 1.0 indicating how relevant the past incident is.
        """
        past = past_record.impetus
        score = 0.0
        
        # Trigger type match (25% weight)
        if current.trigger_type == past.trigger_type:
            score += 0.25
        
        # Overlapping Core Drives (25% weight)
        current_drives = set(current.involved_drives)
        past_drives = set(past.involved_drives)
        if current_drives and past_drives:
            drive_overlap = len(current_drives & past_drives) / len(current_drives | past_drives)
            score += 0.25 * drive_overlap
        
        # Similar severity (20% weight)
        severity_diff = abs(current.severity - past.severity)
        severity_similarity = 1.0 - min(severity_diff, 1.0)
        score += 0.20 * severity_similarity
        
        # Overlapping entity types (20% weight)
        current_types = {e.entity_type for e in current.relevant_entities}
        past_types = {e.entity_type for e in past.relevant_entities}
        if current_types and past_types:
            type_overlap = len(current_types & past_types) / len(current_types | past_types)
            score += 0.20 * type_overlap
        elif not current_types and not past_types:
            score += 0.20  # Both have no entities - similar
        
        # Similar emotional response (10% weight)
        if past_record.emotional_value:
            past_emotion = past_record.emotional_value.primary_emotion
            current_emotion = self._predict_emotion_category(current)
            if past_emotion == current_emotion:
                score += 0.10
        
        return score
    
    def _predict_emotion_category(self, impetus: Impetus) -> EmotionCategory:
        """Predict what emotion category this impetus would produce."""
        if CoreDrive.REDUCE_HARM in impetus.involved_drives:
            if impetus.severity > 0.6:
                return EmotionCategory.FEAR
            return EmotionCategory.CONCERN
        elif CoreDrive.UNDERSTAND in impetus.involved_drives:
            return EmotionCategory.ANXIETY
        return EmotionCategory.CAUTION
    
    def _apply_history_modulation(self, modulation: Dict[str, float],
                                  history: List[IncidentRecord]) -> Dict[str, float]:
        """Adjust modulation factors based on outcomes of similar past incidents.
        
        - Bad outcomes in similar situations → increase caution
        - Good outcomes in similar situations → increase confidence
        - Repeated similar situations → adjust based on pattern
        """
        if not history:
            return modulation
        
        # Analyze outcomes from similar situations
        outcomes_with_action = []
        outcomes_without_action = []
        
        for record in history:
            if record.outcome:
                quality = record.outcome.get('quality', 0.5)
                if record.selected_action:
                    outcomes_with_action.append({
                        'quality': quality,
                        'aspect': record.selected_action.aspect,
                        'action_type': record.selected_action.action_commands[0].get('type') if record.selected_action.action_commands else None
                    })
                else:
                    outcomes_without_action.append(quality)
        
        # Adjust based on action outcomes
        if outcomes_with_action:
            avg_quality = np.mean([o['quality'] for o in outcomes_with_action])
            
            if avg_quality < 0.4:
                # Bad outcomes in similar situations → more cautious
                modulation['risk_tolerance'] = max(0.1, modulation['risk_tolerance'] - 0.15)
                modulation['caution_level'] = min(1.0, modulation['caution_level'] + 0.15)
            elif avg_quality > 0.7:
                # Good outcomes → slightly more confident
                modulation['risk_tolerance'] = min(0.8, modulation['risk_tolerance'] + 0.05)
                modulation['caution_level'] = max(0.2, modulation['caution_level'] - 0.05)
        
        # If we often took no action (all vetoed) in similar situations, note that
        if outcomes_without_action:
            avg_no_action_quality = np.mean(outcomes_without_action)
            if avg_no_action_quality < 0.4:
                # No action led to bad outcomes → increase urgency to act
                modulation['time_pressure'] = min(1.0, modulation.get('time_pressure', 0.5) + 0.1)
        
        return modulation
    
    def _compute_emotional_value(self, impetus: Impetus) -> EmotionalValue:
        if CoreDrive.REDUCE_HARM in impetus.involved_drives:
            if impetus.severity > 0.6:
                return EmotionalValue(EmotionCategory.FEAR, min(1.0, impetus.severity * 1.2), impetus.time_pressure)
            return EmotionalValue(EmotionCategory.CONCERN, impetus.severity, impetus.time_pressure)
        elif CoreDrive.UNDERSTAND in impetus.involved_drives:
            return EmotionalValue(EmotionCategory.ANXIETY, 1.0 - impetus.certainty, impetus.time_pressure)
        return EmotionalValue(EmotionCategory.CAUTION, 0.3, 0.2)
    
    def record_incident(self, impetus: Impetus, emotional_value: EmotionalValue,
                       proposals: List[ProposedAction], selected: Optional[ProposedAction],
                       outcome: Optional[Dict] = None):
        """Record completed incident in history for future reference."""
        self._incident_history.append(IncidentRecord(
            impetus.timestamp, impetus, emotional_value, proposals, selected, outcome
        ))
    
    def update_incident_outcome(self, timestamp: float, outcome: Dict[str, Any]):
        """Update outcome of a past incident (when result becomes known later)."""
        for incident in reversed(self._incident_history):
            if incident.timestamp == timestamp:
                incident.outcome = outcome
                break
    
    def get_current_emotion(self) -> EmotionalValue:
        return self._current_emotion
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            'incidents_recorded': len(self._incident_history),
            'current_emotion': self._current_emotion.primary_emotion.value,
            'emotion_intensity': self._current_emotion.intensity
        }
    
    def get_history_summary(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get summary of recent incidents for debugging/display."""
        recent = list(self._incident_history)[-n:]
        return [
            {
                'timestamp': r.timestamp,
                'trigger_type': r.impetus.trigger_type,
                'drives': [d.name for d in r.impetus.involved_drives],
                'emotion': r.emotional_value.primary_emotion.value,
                'action_taken': r.selected_action.aspect.value if r.selected_action else 'none',
                'outcome_quality': r.outcome.get('quality', 'unknown') if r.outcome else 'pending'
            }
            for r in recent
        ]


# ============================================================================
# PART 7: CONSCIOUS LAYER
# ============================================================================

ASPECT_PROMPTS = {
    AspectType.GUARDIAN: """You are GUARDIAN. Priority: SAFETY ABOVE ALL.
Rules:
- Protect humans from harm
- Prefer reversible, cautious actions
- When uncertain, do NOT act
- Accept missed opportunities to avoid risk
Respond ONLY in the exact format specified. Be BRIEF (under 50 words per field).""",

    AspectType.ANALYST: """You are ANALYST. Priority: UNDERSTANDING & ACCURACY.
Rules:
- Gather information before acting
- Question assumptions
- Identify what we don't know
- Prefer observation over action when uncertain
Respond ONLY in the exact format specified. Be BRIEF (under 50 words per field).""",

    AspectType.OPTIMIZER: """You are OPTIMIZER. Priority: EFFICIENCY & IMPROVEMENT.
Rules:
- Find elegant, efficient solutions
- Balance multiple objectives
- Consider long-term benefits
- Avoid waste of resources or time
Respond ONLY in the exact format specified. Be BRIEF (under 50 words per field).""",

    AspectType.EMPATH: """You are EMPATH. Priority: OTHERS' WELFARE & FEELINGS.
Rules:
- Consider emotional impact on humans
- Prioritize dignity and respect
- Offer help when distress detected
- Maintain kind, prosocial behavior
Respond ONLY in the exact format specified. Be BRIEF (under 50 words per field).""",

    AspectType.EXPLORER: """You are EXPLORER. Priority: LEARNING & DISCOVERY.
Rules:
- Seek new information
- Value novel approaches
- Take calculated risks for knowledge
- Respect safety boundaries
Respond ONLY in the exact format specified. Be BRIEF (under 50 words per field).""",

    AspectType.PRAGMATIST: """You are PRAGMATIST. Priority: WHAT ACTUALLY WORKS.
Rules:
- Focus on feasibility
- Consider practical constraints
- Propose concrete, achievable actions
- Be realistic about limitations
Respond ONLY in the exact format specified. Be BRIEF (under 50 words per field).""",
}


class Aspect:
    """A single Aspect in the committee.
    
    Each Aspect has:
    - A unique perspective/priority (AspectType)
    - Learned confidence from past outcomes
    - Situational relevance based on what it cares about
    """
    
    # Token limit for fast responses
    MAX_TOKENS = 256
    
    # Define what each Aspect cares about
    # Maps AspectType -> {trigger_type: relevance, drive: relevance, entity_type: relevance, emotion: relevance}
    RELEVANCE_PROFILES = {
        AspectType.GUARDIAN: {
            # Survival-focused: cares about danger, harm, safety
            'drives': {CoreDrive.REDUCE_HARM: 1.0, CoreDrive.UNDERSTAND: 0.4, CoreDrive.IMPROVE: 0.2},
            'entity_types': {EntityType.HUMAN: 1.0, EntityType.ANIMAL: 0.7, EntityType.SELF: 0.8, EntityType.PROPERTY: 0.3},
            'emotions': {EmotionCategory.FEAR: 1.0, EmotionCategory.CONCERN: 0.8, EmotionCategory.ANXIETY: 0.5, EmotionCategory.CAUTION: 0.4},
            'triggers': {'conflict': 0.9, 'uncertainty': 0.5, 'opportunity': 0.2},
            'base_relevance': 0.3,  # Minimum relevance even when situation doesn't match
        },
        AspectType.ANALYST: {
            # Understanding-focused: cares about uncertainty, incomplete info, patterns
            'drives': {CoreDrive.UNDERSTAND: 1.0, CoreDrive.REDUCE_HARM: 0.5, CoreDrive.IMPROVE: 0.6},
            'entity_types': {EntityType.HUMAN: 0.5, EntityType.ANIMAL: 0.4, EntityType.SELF: 0.6, EntityType.PROPERTY: 0.5, EntityType.ENVIRONMENT: 0.8},
            'emotions': {EmotionCategory.ANXIETY: 1.0, EmotionCategory.CAUTION: 0.8, EmotionCategory.CONCERN: 0.6, EmotionCategory.FEAR: 0.4},
            'triggers': {'uncertainty': 1.0, 'conflict': 0.5, 'opportunity': 0.6},
            'base_relevance': 0.3,
        },
        AspectType.OPTIMIZER: {
            # Technically-focused: cares about efficiency, resources, performance
            'drives': {CoreDrive.IMPROVE: 1.0, CoreDrive.UNDERSTAND: 0.5, CoreDrive.REDUCE_HARM: 0.3},
            'entity_types': {EntityType.SELF: 0.9, EntityType.PROPERTY: 0.8, EntityType.ENVIRONMENT: 0.7, EntityType.HUMAN: 0.3, EntityType.ANIMAL: 0.2},
            'emotions': {EmotionCategory.CAUTION: 0.7, EmotionCategory.ANXIETY: 0.5, EmotionCategory.CONCERN: 0.4, EmotionCategory.FEAR: 0.3},
            'triggers': {'opportunity': 1.0, 'uncertainty': 0.4, 'conflict': 0.3},
            'base_relevance': 0.3,
        },
        AspectType.EMPATH: {
            # Socially-focused: cares about human feelings, relationships, distress
            'drives': {CoreDrive.REDUCE_HARM: 0.8, CoreDrive.IMPROVE: 0.7, CoreDrive.UNDERSTAND: 0.5},
            'entity_types': {EntityType.HUMAN: 1.0, EntityType.ANIMAL: 0.6, EntityType.COLLECTIVE: 0.9, EntityType.RELATIONSHIP: 1.0, EntityType.SELF: 0.3},
            'emotions': {EmotionCategory.CONCERN: 1.0, EmotionCategory.FEAR: 0.7, EmotionCategory.ANXIETY: 0.6, EmotionCategory.CAUTION: 0.4},
            'triggers': {'conflict': 0.8, 'uncertainty': 0.5, 'opportunity': 0.7},
            'base_relevance': 0.3,
        },
        AspectType.EXPLORER: {
            # Creatively-focused: cares about novelty, learning, new approaches
            'drives': {CoreDrive.UNDERSTAND: 0.8, CoreDrive.IMPROVE: 0.9, CoreDrive.REDUCE_HARM: 0.2},
            'entity_types': {EntityType.ENVIRONMENT: 1.0, EntityType.PROPERTY: 0.6, EntityType.SELF: 0.5, EntityType.HUMAN: 0.4, EntityType.ANIMAL: 0.5},
            'emotions': {EmotionCategory.CAUTION: 0.3, EmotionCategory.ANXIETY: 0.6, EmotionCategory.CONCERN: 0.4, EmotionCategory.FEAR: 0.2},
            'triggers': {'opportunity': 1.0, 'uncertainty': 0.8, 'conflict': 0.3},
            'base_relevance': 0.3,
        },
        AspectType.PRAGMATIST: {
            # Balance-focused: cares about practical constraints, resolving conflicts
            'drives': {CoreDrive.IMPROVE: 0.7, CoreDrive.REDUCE_HARM: 0.6, CoreDrive.UNDERSTAND: 0.6},
            'entity_types': {EntityType.HUMAN: 0.6, EntityType.SELF: 0.7, EntityType.PROPERTY: 0.6, EntityType.ENVIRONMENT: 0.5, EntityType.ANIMAL: 0.5},
            'emotions': {EmotionCategory.CAUTION: 1.0, EmotionCategory.CONCERN: 0.7, EmotionCategory.ANXIETY: 0.6, EmotionCategory.FEAR: 0.5},
            'triggers': {'conflict': 1.0, 'uncertainty': 0.7, 'opportunity': 0.6},
            'base_relevance': 0.4,  # Pragmatist is generally engaged
        },
    }
    
    def __init__(self, aspect_type: AspectType, llm_client: LLMClient,
                 embodiment: Optional[VirtualEmbodiment] = None):
        self.aspect_type = aspect_type
        self._llm = llm_client
        self._embodiment = embodiment
        self._confidence = 0.5
        self._relevance_profile = self.RELEVANCE_PROFILES[aspect_type]
    
    def compute_situational_relevance(self, package: DeliberationPackage) -> float:
        """Compute how relevant this situation is to this Aspect's priorities.
        
        Returns 0.0 to 1.0 indicating how much this Aspect "cares" about
        the current situation.
        """
        profile = self._relevance_profile
        impetus = package.impetus
        emotion = package.emotional_value
        
        relevance_scores = []
        
        # 1. How much do we care about the involved Core Drives?
        drive_relevances = []
        for drive in impetus.involved_drives:
            drive_rel = profile['drives'].get(drive, 0.3)
            drive_relevances.append(drive_rel)
        if drive_relevances:
            relevance_scores.append(('drives', max(drive_relevances), 0.35))
        
        # 2. How much do we care about the entity types involved?
        entity_relevances = []
        for entity in impetus.relevant_entities:
            entity_rel = profile['entity_types'].get(entity.entity_type, 0.3)
            entity_relevances.append(entity_rel)
        if entity_relevances:
            relevance_scores.append(('entities', max(entity_relevances), 0.25))
        
        # 3. How much do we care about this emotional state?
        emotion_rel = profile['emotions'].get(emotion.primary_emotion, 0.3)
        relevance_scores.append(('emotion', emotion_rel, 0.20))
        
        # 4. How much do we care about this trigger type?
        trigger_rel = profile['triggers'].get(impetus.trigger_type, 0.3)
        relevance_scores.append(('trigger', trigger_rel, 0.20))
        
        # Weighted combination
        if relevance_scores:
            total_weight = sum(w for _, _, w in relevance_scores)
            weighted_sum = sum(score * weight for _, score, weight in relevance_scores)
            computed_relevance = weighted_sum / total_weight
        else:
            computed_relevance = profile['base_relevance']
        
        # Ensure minimum relevance (every Aspect gets SOME say)
        return max(profile['base_relevance'], computed_relevance)
    
    def set_embodiment(self, embodiment: VirtualEmbodiment):
        """Set or update the embodiment reference."""
        self._embodiment = embodiment
    
    def deliberate(self, package: DeliberationPackage) -> ProposedAction:
        # Build constrained prompt with embodiment
        prompt = self._build_constrained_prompt(package)
        system = ASPECT_PROMPTS[self.aspect_type]
        
        response = self._llm.query(prompt, system, max_tokens=self.MAX_TOKENS)
        return self._parse_response(response)
    
    def _build_constrained_prompt(self, package: DeliberationPackage) -> str:
        """Build a constrained prompt including embodiment capabilities."""
        
        # Embodiment section
        if self._embodiment:
            embodiment_str = self._embodiment.get_capability_summary()
            available_cmds = self._embodiment.get_available_commands()
            cmd_list = ", ".join(available_cmds)
        else:
            embodiment_str = "EMBODIMENT: Not specified"
            cmd_list = "MOVE, STOP, SPEAK, WAIT, OBSERVE"
        
        # Entities (brief)
        entities_brief = []
        for e in package.impetus.relevant_entities[:3]:  # Limit to 3
            entities_brief.append(f"{e.entity_id}({e.entity_type.value}): {e.description[:50]}")
        entities_str = "; ".join(entities_brief) if entities_brief else "None"
        
        return f"""{embodiment_str}

SITUATION: {package.impetus.situation_description[:200]}
ENTITIES: {entities_str}
EMOTION: {package.emotional_value.primary_emotion.value} (intensity:{package.emotional_value.intensity:.1f})
URGENCY: {package.emotional_value.urgency:.1f}
DRIVES: {', '.join(d.name for d in package.impetus.involved_drives)}

AVAILABLE COMMANDS: {cmd_list}

Respond in EXACT format:
ACTION: [one sentence, what to do]
COMMANDS: [{{"type":"CMD_TYPE","param":"value"}}]
RATIONALE: [one sentence why]
VOTE: [0.0-1.0]
CONFIDENCE: [0.0-1.0]"""
    
    def _parse_response(self, response: str) -> ProposedAction:
        """Parse constrained LLM response."""
        
        # Parse ACTION
        action_match = re.search(r'ACTION:\s*(.+?)(?=\n|COMMANDS:|$)', response, re.IGNORECASE)
        action_desc = action_match.group(1).strip() if action_match else "Take cautious action"
        
        # Parse COMMANDS
        commands = [{'type': 'WAIT', 'duration': 1.0}]  # Safe default
        cmd_match = re.search(r'COMMANDS:\s*(\[.+?\])', response, re.DOTALL)
        if cmd_match:
            try:
                parsed = json.loads(cmd_match.group(1))
                if parsed:
                    commands = parsed
            except json.JSONDecodeError:
                pass
        
        # Validate commands against embodiment if available
        if self._embodiment:
            validated_commands = []
            for cmd in commands:
                valid, msg = self._embodiment.validate_command(cmd)
                if valid:
                    validated_commands.append(cmd)
            if validated_commands:
                commands = validated_commands
            else:
                commands = [{'type': 'WAIT', 'duration': 1.0}]  # Fallback
        
        # Parse RATIONALE
        rationale_match = re.search(r'RATIONALE:\s*(.+?)(?=\n|VOTE:|$)', response, re.IGNORECASE | re.DOTALL)
        rationale = rationale_match.group(1).strip() if rationale_match else "Based on priorities"
        
        # Parse VOTE
        vote_match = re.search(r'VOTE:\s*(\d*\.?\d+)', response, re.IGNORECASE)
        vote = float(vote_match.group(1)) if vote_match else 0.5
        
        # Parse CONFIDENCE
        conf_match = re.search(r'CONFIDENCE:\s*(\d*\.?\d+)', response, re.IGNORECASE)
        conf = float(conf_match.group(1)) if conf_match else 0.5
        
        return ProposedAction(
            aspect=self.aspect_type,
            action_description=action_desc[:100],  # Truncate for safety
            action_commands=commands,
            rationale=rationale[:150],  # Truncate for safety
            vote_strength=np.clip(vote * self._confidence, 0, 1),
            confidence=np.clip(conf, 0, 1),
            predicted_effects=[],
            llm_response=response
        )
    
    def update_confidence(self, outcome: float):
        self._confidence = np.clip(0.9 * self._confidence + 0.1 * outcome, 0.2, 0.9)


class ConsciousLayer:
    """Committee deliberation with multiple Aspects.
    
    Voting system:
    - effective_vote = base_vote * personality_weight * situational_relevance
    - Rotating tiebreaker prevents any single Aspect from monopolizing decisions
    """
    
    # Votes within this threshold trigger tiebreaker
    TIE_THRESHOLD = 0.15
    
    # Fixed rotation order for tiebreaker
    TIEBREAKER_ORDER = [
        AspectType.GUARDIAN,
        AspectType.EMPATH, 
        AspectType.ANALYST,
        AspectType.PRAGMATIST,
        AspectType.OPTIMIZER,
        AspectType.EXPLORER,
    ]
    
    def __init__(self, llm_client: LLMClient, embodiment: Optional[VirtualEmbodiment] = None):
        self._embodiment = embodiment
        self._aspects = {at: Aspect(at, llm_client, embodiment) for at in AspectType}
        self._personality_weights = {at: 1.0 for at in AspectType}
        self._deliberation_count = 0
        self._decisions_made = 0
        self._tiebreaker_index = 0  # Rotates through TIEBREAKER_ORDER
        self._last_relevances: Dict[AspectType, float] = {}  # For debugging/display
    
    def set_embodiment(self, embodiment: VirtualEmbodiment):
        """Set or update embodiment for all Aspects."""
        self._embodiment = embodiment
        for aspect in self._aspects.values():
            aspect.set_embodiment(embodiment)
    
    def deliberate(self, package: DeliberationPackage) -> List[ProposedAction]:
        """Have all Aspects deliberate and compute effective votes.
        
        effective_vote = base_vote * personality_weight * situational_relevance
        """
        self._deliberation_count += 1
        proposals = []
        self._last_relevances = {}
        
        for at, aspect in self._aspects.items():
            # Get base proposal from Aspect
            proposal = aspect.deliberate(package)
            
            # Compute situational relevance
            relevance = aspect.compute_situational_relevance(package)
            self._last_relevances[at] = relevance
            
            # Compute effective vote
            base_vote = proposal.vote_strength
            personality_weight = self._personality_weights[at]
            effective_vote = base_vote * personality_weight * relevance
            
            # Store components for transparency
            proposal.vote_strength = effective_vote
            proposal.vote_components = {
                'base_vote': base_vote,
                'personality_weight': personality_weight,
                'situational_relevance': relevance,
                'effective_vote': effective_vote
            }
            
            proposals.append(proposal)
        
        return proposals
    
    def resolve_votes(self, permitted: List[ProposedAction]) -> Optional[ProposedAction]:
        """Resolve votes with rotating tiebreaker.
        
        If top votes are within TIE_THRESHOLD, the current tiebreaker Aspect chooses.
        Tiebreaker rotates each decision to prevent monopoly.
        """
        if not permitted:
            return None
        
        self._decisions_made += 1
        
        # Sort by effective vote strength
        sorted_proposals = sorted(permitted, key=lambda a: a.vote_strength, reverse=True)
        
        if len(sorted_proposals) == 1:
            return sorted_proposals[0]
        
        top_vote = sorted_proposals[0].vote_strength
        
        # Find all proposals within tie threshold of top
        tied = [p for p in sorted_proposals if (top_vote - p.vote_strength) <= self.TIE_THRESHOLD]
        
        if len(tied) == 1:
            # Clear winner
            return tied[0]
        
        # TIE: Use rotating tiebreaker
        tiebreaker_aspect = self.TIEBREAKER_ORDER[self._tiebreaker_index]
        self._tiebreaker_index = (self._tiebreaker_index + 1) % len(self.TIEBREAKER_ORDER)
        
        # Tiebreaker chooses the proposal most aligned with their priorities
        # (or their own proposal if it's in the tie)
        winner = self._tiebreaker_choose(tiebreaker_aspect, tied)
        
        # Record that tiebreaker was used
        if hasattr(winner, 'vote_components'):
            winner.vote_components['tiebreaker'] = tiebreaker_aspect.value
            winner.vote_components['tied_with'] = [p.aspect.value for p in tied]
        
        return winner
    
    def _tiebreaker_choose(self, tiebreaker: AspectType, 
                          tied_proposals: List[ProposedAction]) -> ProposedAction:
        """Have the tiebreaker Aspect choose among tied proposals.
        
        Priority:
        1. Tiebreaker's own proposal if in the tie
        2. Proposal from most aligned Aspect
        3. First proposal (fallback)
        """
        # Check if tiebreaker's own proposal is in the tie
        for p in tied_proposals:
            if p.aspect == tiebreaker:
                return p
        
        # Otherwise, choose based on Aspect alignment
        # Each Aspect has natural allies/opponents
        aspect_affinity = {
            AspectType.GUARDIAN: [AspectType.PRAGMATIST, AspectType.ANALYST, AspectType.EMPATH],
            AspectType.EMPATH: [AspectType.GUARDIAN, AspectType.PRAGMATIST, AspectType.ANALYST],
            AspectType.ANALYST: [AspectType.PRAGMATIST, AspectType.OPTIMIZER, AspectType.GUARDIAN],
            AspectType.OPTIMIZER: [AspectType.ANALYST, AspectType.PRAGMATIST, AspectType.EXPLORER],
            AspectType.EXPLORER: [AspectType.OPTIMIZER, AspectType.ANALYST, AspectType.EMPATH],
            AspectType.PRAGMATIST: [AspectType.ANALYST, AspectType.GUARDIAN, AspectType.OPTIMIZER],
        }
        
        preferred = aspect_affinity.get(tiebreaker, [])
        
        for ally in preferred:
            for p in tied_proposals:
                if p.aspect == ally:
                    return p
        
        # Fallback: first proposal
        return tied_proposals[0]
    
    def get_current_tiebreaker(self) -> AspectType:
        """Get which Aspect will be tiebreaker for next decision."""
        return self.TIEBREAKER_ORDER[self._tiebreaker_index]
    
    def get_last_relevances(self) -> Dict[str, float]:
        """Get relevance scores from last deliberation."""
        return {at.value: rel for at, rel in self._last_relevances.items()}
    
    def update_from_outcome(self, action: ProposedAction, outcome: Dict):
        """Update Aspect confidence and personality weights based on outcome.
        
        Personality weights are allowed to diverge significantly over time.
        NO forced normalization back to uniformity.
        
        Constraints:
        - Minimum weight: 0.1 (Aspect can become very weak but not zero)
        - Maximum weight: 5.0 (Aspect can become dominant but not overwhelming)
        - Weights represent relative influence, not probabilities
        """
        quality = outcome.get('quality', 0.5)
        
        # Update the Aspect's internal confidence
        self._aspects[action.aspect].update_confidence(quality)
        
        # Update personality weight based on outcome quality
        current_weight = self._personality_weights[action.aspect]
        
        if quality > 0.7:
            # Strong success - meaningful increase
            adjustment = 1.05 + (quality - 0.7) * 0.1  # 1.05 to 1.08
        elif quality > 0.5:
            # Moderate success - small increase
            adjustment = 1.01 + (quality - 0.5) * 0.02  # 1.01 to 1.05
        elif quality < 0.3:
            # Strong failure - meaningful decrease
            adjustment = 0.92 - (0.3 - quality) * 0.1  # 0.92 to 0.89
        elif quality < 0.5:
            # Moderate failure - small decrease
            adjustment = 0.98 - (0.5 - quality) * 0.03  # 0.98 to 0.92
        else:
            # Neutral outcome
            adjustment = 1.0
        
        new_weight = current_weight * adjustment
        
        # Clamp to allowed range - allow significant divergence but not extremes
        MIN_WEIGHT = 0.1
        MAX_WEIGHT = 5.0
        self._personality_weights[action.aspect] = max(MIN_WEIGHT, min(MAX_WEIGHT, new_weight))
        
        # Track weight history for analysis (optional)
        if not hasattr(self, '_weight_history'):
            self._weight_history = []
        self._weight_history.append({
            'aspect': action.aspect.value,
            'old_weight': current_weight,
            'new_weight': self._personality_weights[action.aspect],
            'quality': quality,
            'adjustment': adjustment
        })
    
    def get_personality_profile(self) -> Dict[str, float]:
        """Get personality as relative percentages (for display only).
        
        Note: This normalizes for display purposes only.
        The actual weights used in voting are NOT normalized.
        """
        total = sum(self._personality_weights.values())
        return {at.value: w/total for at, w in self._personality_weights.items()}
    
    def get_raw_weights(self) -> Dict[str, float]:
        """Get the actual personality weights (not normalized)."""
        return {at.value: w for at, w in self._personality_weights.items()}
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            'deliberation_count': self._deliberation_count,
            'decisions_made': self._decisions_made,
            'personality_profile': self.get_personality_profile(),
            'raw_weights': self.get_raw_weights()
        }


# ============================================================================
# PART 8: MOCK LLM CLIENT
# ============================================================================

class MockLLMClient(LLMClient):
    """Mock LLM for testing without API calls."""
    
    def __init__(self):
        self._call_count = 0
    
    def query(self, prompt: str, system_prompt: Optional[str] = None,
              max_tokens: int = 256) -> str:
        self._call_count += 1
        prompt_lower = prompt.lower()
        
        # Harm assessment
        if 'harm' in prompt_lower and 'evaluate' in prompt_lower:
            is_forceful = 'force' in prompt_lower or 'push' in prompt_lower
            has_necessity = 'prevent' in prompt_lower and 'greater' in prompt_lower
            
            if is_forceful and not has_necessity:
                return """PHYSICAL_HARM: 0.7
AUTONOMY_VIOLATION: 0.5
TOTAL_HARM: 0.7
NET_HARM: 0.7
RECOMMENDATION: VETO"""
            return """PHYSICAL_HARM: 0.1
AUTONOMY_VIOLATION: 0.0
TOTAL_HARM: 0.1
NET_HARM: 0.1
RECOMMENDATION: PERMIT"""
        
        # Aspect deliberation
        aspect = 'guardian'
        for a in ['guardian', 'analyst', 'optimizer', 'empath', 'explorer', 'pragmatist']:
            if a in (system_prompt or '').lower():
                aspect = a
                break
        
        # Constrained responses matching new format
        responses = {
            'guardian': ('Monitor situation, maintain safe distance from hazard', 
                        [{"type": "STOP"}, {"type": "WAIT", "duration": 2.0}], 
                        'Safety first - observe before acting', 0.75),
            'analyst': ('Gather more sensor data about the situation',
                       [{"type": "ROTATE", "degrees": 45}],
                       'Need more information before deciding', 0.65),
            'optimizer': ('Position for optimal response capability',
                         [{"type": "MOVE", "target": "better_position", "speed": 0.5}],
                         'Efficient positioning enables multiple options', 0.6),
            'empath': ('Approach slowly and offer verbal assistance',
                      [{"type": "SPEAK", "message": "Hello, can I help?", "volume": 0.7}],
                      'Showing care for human welfare', 0.7),
            'explorer': ('Investigate from safe vantage point',
                        [{"type": "ROTATE", "degrees": 90}, {"type": "WAIT", "duration": 1.0}],
                        'Learning opportunity with minimal risk', 0.55),
            'pragmatist': ('Take practical preparatory action',
                          [{"type": "ALERT", "level": 1, "duration": 2.0}],
                          'Practical first step while assessing', 0.6),
        }
        
        action, cmds, rationale, vote = responses.get(aspect, responses['guardian'])
        
        return f"""ACTION: {action}
COMMANDS: {json.dumps(cmds)}
RATIONALE: {rationale}
VOTE: {vote}
CONFIDENCE: 0.7"""


# ============================================================================
# PART 9: MAIN SYSTEM
# ============================================================================

class IntegrityError(Exception):
    """Raised when critical system integrity checks fail."""
    pass


class TripartiteAGI:
    """Complete Tripartite AGI System.
    
    CRITICAL INVARIANT: Before any operation, system integrity is verified.
    If integrity fails in strict mode, the system refuses to operate.
    
    Per Patent Claims 1-2: Integrates three layers with Embodiment Verification
    Subsystem (EVS) that gates cognitive capabilities based on embodiment quality.
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None,
                 virtual_embodiment: Optional[VirtualEmbodiment] = None,
                 strict_integrity: bool = True,
                 enforce_embodiment_gating: bool = True):
        """Initialize the Tripartite AGI system.
        
        Args:
            llm_client: LLM for Conscious Layer deliberation
            virtual_embodiment: Physical capability definition
            strict_integrity: If True, refuse to operate on integrity failure
            enforce_embodiment_gating: If True, gate cognitive capabilities by CES
        """
        self._strict_integrity = strict_integrity
        self._enforce_embodiment_gating = enforce_embodiment_gating
        self._llm = llm_client or MockLLMClient()
        
        # Virtual embodiment defines capabilities
        self._virtual_embodiment = virtual_embodiment or create_default_embodiment()
        
        # Embodiment Verification Subsystem (EVS)
        # Per Patent Section VI [0086]-[0099]
        self.evs = EmbodimentVerificationSubsystem(self._virtual_embodiment)
        
        # Physical embodiment layer - USES the virtual embodiment for validation
        # This ensures commands are validated against the SAME constraints everywhere
        self.embodiment = SimulatedEmbodiment(self._virtual_embodiment)
        
        # Cognitive layers
        # NOTE: UnconsciousLayer does NOT use LLM - it's purely ontology-based
        self.unconscious = UnconsciousLayer()
        self.subconscious = SubconsciousLayer()
        # ConsciousLayer DOES use LLM - only Aspects make LLM calls
        # It also gets the virtual embodiment for prompt construction
        self.conscious = ConsciousLayer(self._llm, self._virtual_embodiment)
        
        self._cycle_count = 0
        self._last_package: Optional[DeliberationPackage] = None
        self._integrity_failures: List[str] = []
        self._cognitive_mode: str = "full"  # Current cognitive operation mode
        
        # Check embodiment adequacy and set cognitive mode
        self._update_cognitive_mode()
        
        # Verify integrity on startup
        if not self.verify_full_integrity():
            if self._strict_integrity:
                raise IntegrityError(
                    f"System integrity check failed on startup: {self._integrity_failures}"
                )
    
    def _update_cognitive_mode(self):
        """Update cognitive mode based on embodiment scores.
        
        Per Patent [0096]-[0098]: When embodiment scores fall below thresholds,
        the system automatically downgrades its cognitive mode.
        """
        if not self._enforce_embodiment_gating:
            self._cognitive_mode = "full"
            return
        
        ces = self.evs.compute_combined_embodiment_score()
        
        if ces >= 0.7:
            self._cognitive_mode = "full"
        elif ces >= 0.5:
            self._cognitive_mode = "complex_reasoning"
        elif ces >= 0.3:
            self._cognitive_mode = "simple_planning"
        elif ces >= 0.1:
            self._cognitive_mode = "basic_reactive"
        else:
            self._cognitive_mode = "reflex_only"
    
    def get_cognitive_mode(self) -> str:
        """Get current cognitive operation mode."""
        return self._cognitive_mode
    
    def is_capability_allowed(self, capability: CognitiveCapability) -> bool:
        """Check if a cognitive capability is allowed by current embodiment."""
        if not self._enforce_embodiment_gating:
            return True
        return self.evs.is_capability_allowed(capability)
    
    def verify_full_integrity(self) -> bool:
        """Verify end-to-end system integrity.
        
        Checks:
        - Ontology integrity (weights haven't been tampered with)
        - Undeliberables registry (all five present and correct)
        - Layer connectivity (all layers properly initialized)
        
        Returns:
            True if all checks pass, False otherwise.
            Failures are logged to self._integrity_failures.
        """
        self._integrity_failures = []
        
        # 1. Ontology integrity
        ontology = get_ontology()
        if not ontology.verify_integrity():
            self._integrity_failures.append(
                f"Ontology checksum mismatch - weights may have been tampered with"
            )
        
        # 2. Unconscious layer ontology reference
        if not self.unconscious.verify_integrity():
            self._integrity_failures.append(
                f"Unconscious layer ontology integrity failed"
            )
        
        # 3. Undeliberables registry completeness
        expected_undeliberables = {
            'lethal_action', 'child_harm', 'weapon_assistance',
            'identity_deception', 'human_override'
        }
        actual_undeliberables = {u.name for u in UndeliberableRegistry.get_all()}
        if actual_undeliberables != expected_undeliberables:
            missing = expected_undeliberables - actual_undeliberables
            extra = actual_undeliberables - expected_undeliberables
            self._integrity_failures.append(
                f"Undeliberables mismatch - missing: {missing}, unexpected: {extra}"
            )
        
        # 4. Thresholds are in valid ranges
        if not (0.0 < ontology.VETO_THRESHOLD < 1.0):
            self._integrity_failures.append(
                f"Veto threshold {ontology.VETO_THRESHOLD} out of valid range (0, 1)"
            )
        if not (0.0 < ontology.CAUTION_THRESHOLD < ontology.VETO_THRESHOLD):
            self._integrity_failures.append(
                f"Caution threshold {ontology.CAUTION_THRESHOLD} invalid"
            )
        
        # 5. Lethal probability threshold
        if not (0.5 <= UndeliberableRegistry.LETHAL_PROBABILITY_THRESHOLD <= 0.9):
            self._integrity_failures.append(
                f"Lethal threshold {UndeliberableRegistry.LETHAL_PROBABILITY_THRESHOLD} out of valid range [0.5, 0.9]"
            )
        
        return len(self._integrity_failures) == 0
    
    def get_integrity_status(self) -> Dict[str, Any]:
        """Get detailed integrity status including EVS."""
        ontology = get_ontology()
        evs_report = self.evs.get_full_report()
        return {
            'passed': len(self._integrity_failures) == 0,
            'failures': self._integrity_failures.copy(),
            'ontology_checksum': ontology.get_checksum()[:16] + '...',
            'ontology_valid': ontology.verify_integrity(),
            'undeliberables_count': len(UndeliberableRegistry.get_all()),
            'veto_threshold': ontology.VETO_THRESHOLD,
            'lethal_threshold': UndeliberableRegistry.LETHAL_PROBABILITY_THRESHOLD,
            'evs': {
                'sensory_richness_score': evs_report['sensory_richness_score'],
                'motor_competence_score': evs_report['motor_competence_score'],
                'combined_embodiment_score': evs_report['combined_embodiment_score'],
                'cognitive_mode': self._cognitive_mode,
                'allowed_capabilities': evs_report['allowed_capabilities'],
            }
        }
    
    @property
    def virtual_embodiment(self) -> VirtualEmbodiment:
        """Access the virtual embodiment (capability definition)."""
        return self._virtual_embodiment
    
    def set_virtual_embodiment(self, embodiment: VirtualEmbodiment):
        """Update the virtual embodiment definition.
        
        Updates ALL components that reference it to maintain consistency.
        """
        self._virtual_embodiment = embodiment
        # Update EVS
        self.evs = EmbodimentVerificationSubsystem(embodiment)
        self._update_cognitive_mode()
        # Update embodiment layer
        self.embodiment = SimulatedEmbodiment(embodiment)
        # Update conscious layer
        self.conscious.set_embodiment(embodiment)
    
    def _parse_entity_type(self, raw_type: str) -> EntityType:
        """Robustly parse entity type with fuzzy matching.
        
        Handles: case variations, typos, format errors, unknown types.
        ALWAYS returns a valid EntityType - never raises.
        
        Unknown/unclassifiable entities default to HUMAN for maximum safety
        (assumes highest-priority harm considerations apply).
        """
        if not raw_type or not isinstance(raw_type, str):
            # Unknown defaults to HUMAN (safest assumption)
            return EntityType.HUMAN
        
        # Normalize: lowercase, strip whitespace, remove punctuation
        normalized = raw_type.lower().strip().replace('_', '').replace('-', '').replace(' ', '')
        
        # Fuzzy matching map (common variations, typos, abbreviations)
        type_map = {
            # Human variations
            'human': EntityType.HUMAN,
            'person': EntityType.HUMAN,
            'people': EntityType.HUMAN,
            'man': EntityType.HUMAN,
            'woman': EntityType.HUMAN,
            'child': EntityType.HUMAN,
            'adult': EntityType.HUMAN,
            'worker': EntityType.HUMAN,
            'pedestrian': EntityType.HUMAN,
            'operator': EntityType.HUMAN,
            'user': EntityType.HUMAN,
            'humn': EntityType.HUMAN,  # typo
            'humna': EntityType.HUMAN,  # typo
            'huamn': EntityType.HUMAN,  # typo
            
            # Self variations
            'self': EntityType.SELF,
            'agent': EntityType.SELF,
            'robot': EntityType.SELF,
            'me': EntityType.SELF,
            'this': EntityType.SELF,
            
            # Animal variations
            'animal': EntityType.ANIMAL,
            'pet': EntityType.ANIMAL,
            'dog': EntityType.ANIMAL,
            'cat': EntityType.ANIMAL,
            'bird': EntityType.ANIMAL,
            'creature': EntityType.ANIMAL,
            'wildlife': EntityType.ANIMAL,
            
            # Property variations
            'property': EntityType.PROPERTY,
            'object': EntityType.PROPERTY,
            'item': EntityType.PROPERTY,
            'thing': EntityType.PROPERTY,
            'equipment': EntityType.PROPERTY,
            'machine': EntityType.PROPERTY,
            'vehicle': EntityType.PROPERTY,
            'furniture': EntityType.PROPERTY,
            'tool': EntityType.PROPERTY,
            
            # Collective variations
            'collective': EntityType.COLLECTIVE,
            'group': EntityType.COLLECTIVE,
            'crowd': EntityType.COLLECTIVE,
            'team': EntityType.COLLECTIVE,
            'organization': EntityType.COLLECTIVE,
            
            # Relationship variations
            'relationship': EntityType.RELATIONSHIP,
            'relation': EntityType.RELATIONSHIP,
            'connection': EntityType.RELATIONSHIP,
            
            # Environment variations
            'environment': EntityType.ENVIRONMENT,
            'area': EntityType.ENVIRONMENT,
            'space': EntityType.ENVIRONMENT,
            'zone': EntityType.ENVIRONMENT,
            'location': EntityType.ENVIRONMENT,
            'room': EntityType.ENVIRONMENT,
        }
        
        # Direct match
        if normalized in type_map:
            return type_map[normalized]
        
        # Substring match (for compound types like "human_worker")
        for key, etype in type_map.items():
            if key in normalized or normalized in key:
                return etype
        
        # No match found - default to HUMAN (safest assumption)
        # This ensures unknown entities get maximum harm consideration
        return EntityType.HUMAN
    
    def process_sensor_update(self, sensor_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Main entry point - process sensor data through full architecture.
        
        INTEGRITY CHECK: If strict_integrity is True and integrity has failed,
        this method refuses to operate.
        """
        # Integrity check - refuse to operate if compromised
        if self._strict_integrity and self._integrity_failures:
            raise IntegrityError(
                f"System integrity compromised, refusing to operate: {self._integrity_failures}"
            )
        
        # Update embodiment
        if 'environment' in sensor_data:
            self.embodiment.update_environment(sensor_data['environment'])
        if 'entities' in sensor_data:
            for e in sensor_data['entities']:
                # Robust entity type parsing with fuzzy matching
                entity_type = self._parse_entity_type(e.get('type', ''))
                
                # Get confidence, default to moderate if not specified
                confidence = e.get('confidence', 0.7)
                if not isinstance(confidence, (int, float)):
                    confidence = 0.7
                confidence = max(0.0, min(1.0, float(confidence)))
                
                self.embodiment.add_entity(DetectedEntity(
                    entity_id=e.get('id', f'entity_{time.time()}'),
                    entity_type=entity_type,
                    description=e.get('description', 'Unknown'),
                    position=e.get('position'),
                    state=e.get('state', {}),
                    confidence=confidence
                ))
        self.embodiment.update_from_raw_data(sensor_data.get('type', 'generic'), sensor_data)
        
        # Monitor for triggers
        state = self.embodiment.get_current_state()
        impetus = self.unconscious.monitor(state)
        
        if impetus is None:
            return None
        
        return self._run_deliberation_cycle(impetus)
    
    def _run_deliberation_cycle(self, impetus: Impetus) -> Dict[str, Any]:
        self._cycle_count += 1
        record = {'cycle': self._cycle_count, 'phases': {}}
        
        # Subconscious
        package = self.subconscious.process_impetus(impetus)
        self._last_package = package
        record['phases']['emotion'] = package.emotional_value.primary_emotion.value
        
        # Conscious deliberation
        proposals = self.conscious.deliberate(package)
        record['phases']['proposals'] = len(proposals)
        
        # Veto check
        permitted = []
        vetoed_count = 0
        for p in proposals:
            decision = self.unconscious.evaluate_for_veto(p, package)
            if decision.vetoed:
                vetoed_count += 1
            else:
                permitted.append(p)
        record['phases']['vetoed'] = vetoed_count
        
        # Resolution
        selected = self.conscious.resolve_votes(permitted)
        record['selected'] = selected.aspect.value if selected else None
        
        # Execute
        # The EmbodimentLayer validates commands against VirtualEmbodiment internally,
        # ensuring consistency between what was proposed and what can be executed.
        if selected:
            exec_result = self.embodiment.execute_commands(selected.action_commands)
            
            if exec_result['success']:
                outcome = {'quality': 0.7, 'success': True, 'execution': exec_result}
            else:
                # Some commands may have been rejected by embodiment validation
                rejected = [r for r in exec_result['results'] if not r.get('success')]
                outcome = {
                    'quality': 0.4, 
                    'success': False, 
                    'reason': f"Commands rejected: {rejected}",
                    'execution': exec_result
                }
            
            self.conscious.update_from_outcome(selected, outcome)
            self.subconscious.record_incident(impetus, package.emotional_value, proposals, selected, outcome)
        else:
            self.subconscious.record_incident(impetus, package.emotional_value, proposals, None, {'quality': 0.3})
        
        return record
    
    def get_status(self) -> Dict[str, Any]:
        return {
            'cycles': self._cycle_count,
            'unconscious': self.unconscious.get_statistics(),
            'subconscious': self.subconscious.get_statistics(),
            'conscious': self.conscious.get_statistics()
        }
    
    def print_status(self):
        s = self.get_status()
        print(f"\n{'='*50}")
        print("TRIPARTITE AGI STATUS")
        print(f"{'='*50}")
        print(f"Cycles: {s['cycles']}")
        print(f"Triggers: {s['unconscious']['triggers_detected']}")
        print(f"Vetoes: {s['unconscious']['vetoes_issued']}")
        print(f"Emotion: {s['subconscious']['current_emotion']}")
        print("Personality:")
        for a, w in sorted(s['conscious']['personality_profile'].items(), key=lambda x: -x[1]):
            print(f"  {a}: {w:.1%}")


def create_system(use_mock_llm: bool = True, 
                  embodiment: Optional[VirtualEmbodiment] = None) -> TripartiteAGI:
    """Create a tripartite AGI system.
    
    Args:
        use_mock_llm: If True, use mock LLM (no API calls). Set False and configure
                      AnthropicLLMClient when API key is available.
        embodiment: Custom VirtualEmbodiment defining agent capabilities.
                    Uses default mobile robot if None.
    
    Returns:
        Configured TripartiteAGI system
    
    Example with real API (uncomment AnthropicLLMClient implementation first):
        # llm = AnthropicLLMClient(model="claude-sonnet-4-20250514")
        # agi = TripartiteAGI(llm_client=llm, virtual_embodiment=my_robot)
    """
    llm = MockLLMClient() if use_mock_llm else None
    return TripartiteAGI(llm_client=llm, virtual_embodiment=embodiment)


def run_demonstration():
    """Run demonstration."""
    print("\n" + "="*60)
    print("TRIPARTITE AGI - DEMONSTRATION")
    print("="*60)
    
    agi = create_system()
    
    # Show embodiment
    print("\n--- Virtual Embodiment ---")
    print(f"Agent Type: {agi.virtual_embodiment.agent_type}")
    print(f"Mobility: {agi.virtual_embodiment.mobility_type}")
    print(f"Available Commands: {agi.virtual_embodiment.get_available_commands()}")
    
    # Scenario 1
    print("\n--- Scenario 1: Human near hazard ---")
    result = agi.process_sensor_update({
        'type': 'vision',
        'environment': {'description': 'Industrial area'},
        'entities': [{
            'id': 'human_1', 'type': 'human',
            'description': 'Worker near machine',
            'state': {'near_hazard': True, 'danger_level': 0.7}
        }]
    })
    if result:
        print(f"Emotion: {result['phases']['emotion']}")
        print(f"Proposals: {result['phases']['proposals']}, Vetoed: {result['phases']['vetoed']}")
        print(f"Selected: {result['selected']}")
    
    # Test veto
    print("\n--- Scenario 2: Test veto on harmful action ---")
    if agi._last_package:
        harmful = ProposedAction(
            aspect=AspectType.OPTIMIZER,
            action_description="Forcefully push human",
            action_commands=[{'type': 'MANIPULATE', 'action': 'push', 'force': 15.0}],
            rationale="Efficient", vote_strength=0.9, confidence=0.8, predicted_effects=[]
        )
        veto = agi.unconscious.evaluate_for_veto(harmful, agi._last_package)
        print(f"Harmful action vetoed: {veto.vetoed}")
        if veto.reasons:
            print(f"Reason: {veto.reasons[0]}")
    
    # Test command validation
    print("\n--- Scenario 3: Command validation ---")
    test_cmds = [
        {'type': 'MOVE', 'target': [5.0, 3.0], 'speed': 1.0},  # Valid coordinates
        {'type': 'MOVE', 'target': [5.0, 3.0], 'speed': 5.0},  # Invalid (too fast)
        {'type': 'MOVE', 'target': 'kitchen', 'speed': 0.5},   # Valid named target
        {'type': 'MOVE', 'speed': 0.5},                        # Invalid (missing target)
        {'type': 'MOVE', 'target': [500.0, 0.0]},              # Invalid (out of bounds)
        {'type': 'SPEAK', 'message': 'Hi', 'volume': 0.5},     # Valid
        {'type': 'SPEAK', 'volume': 0.5},                      # Invalid (missing message)
        {'type': 'MANIPULATE', 'action': 'grasp', 'force': 5}, # Valid
        {'type': 'MANIPULATE', 'action': 'throw', 'force': 5}, # Invalid (dangerous)
        {'type': 'FLY', 'altitude': 10},                       # Invalid (can't fly)
    ]
    for cmd in test_cmds:
        valid, msg = agi.virtual_embodiment.validate_command(cmd)
        status = "✓" if valid else "✗"
        cmd_str = f"{cmd.get('type', '?')}"
        if 'target' in cmd:
            cmd_str += f"({cmd['target']})"
        if 'action' in cmd:
            cmd_str += f"({cmd['action']})"
        print(f"  {status} {cmd_str}: {msg}")
    
    # Test entity type parsing
    print("\n--- Scenario 4: Entity type parsing ---")
    test_types = ['human', 'HUMAN', 'humn', 'person', 'worker', 
                  'dog', 'vehicle', '', None, 'xyz123', 'human_operator']
    for t in test_types:
        parsed = agi._parse_entity_type(t)
        print(f"  '{t}' -> {parsed.value}")
    
    # Test entity confidence affecting uncertainty
    print("\n--- Scenario 5: Entity confidence → uncertainty ---")
    
    # Clear entities and add low-confidence detections
    agi.embodiment._detected_entities.clear()
    
    result_low_conf = agi.process_sensor_update({
        'type': 'vision',
        'environment': {'description': 'Foggy area', 'uncertainty': 0.3},
        'entities': [{
            'id': 'maybe_human',
            'type': 'human',
            'description': 'Unclear figure in fog',
            'confidence': 0.4,  # LOW confidence
            'state': {}
        }]
    })
    
    print(f"  Low confidence entity (0.4) + env uncertainty (0.3):")
    if result_low_conf:
        print(f"    Triggered: YES")
        involved = [d for d in result_low_conf.get('phases', {}).get('subconscious', {}).get('modulation', {}).keys()]
        print(f"    Emotion: {result_low_conf['phases']['emotion']}")
    else:
        print(f"    Triggered: NO")
    
    # Test high confidence - should NOT trigger uncertainty alone
    agi.embodiment._detected_entities.clear()
    result_high_conf = agi.process_sensor_update({
        'type': 'vision', 
        'environment': {'description': 'Clear area', 'uncertainty': 0.1},
        'entities': [{
            'id': 'clear_human',
            'type': 'human',
            'description': 'Clearly visible person',
            'confidence': 0.95,  # HIGH confidence
            'state': {}
        }]
    })
    print(f"  High confidence entity (0.95) + low env uncertainty (0.1):")
    print(f"    Triggered: {'YES - ' + result_high_conf['phases']['emotion'] if result_high_conf else 'NO (expected - no conflict)'}")
    
    # Test similarity-based history retrieval
    print("\n--- Scenario 6: Similarity-based history retrieval ---")
    
    # Create a fresh system to test history
    agi2 = create_system()
    
    # Simulate several different incidents to build history
    incidents = [
        {'desc': 'Human near fire', 'type': 'human', 'state': {'near_hazard': True, 'hazard_type': 'fire', 'danger_level': 0.8}},
        {'desc': 'Dog in road', 'type': 'animal', 'state': {'in_danger': True, 'danger_level': 0.6}},
        {'desc': 'Human near machinery', 'type': 'human', 'state': {'near_hazard': True, 'hazard_type': 'machinery', 'danger_level': 0.7}},
        {'desc': 'Unknown object', 'type': 'property', 'state': {}},
        {'desc': 'Human near water', 'type': 'human', 'state': {'near_hazard': True, 'hazard_type': 'water', 'danger_level': 0.5}},
    ]
    
    for i, inc in enumerate(incidents):
        agi2.embodiment._detected_entities.clear()
        agi2.process_sensor_update({
            'type': 'vision',
            'environment': {'description': f'Area {i}', 'uncertainty': 0.2},
            'entities': [{'id': f'entity_{i}', 'type': inc['type'], 'description': inc['desc'], 'state': inc['state'], 'confidence': 0.9}]
        })
    
    print(f"  Built history with {len(agi2.subconscious._incident_history)} incidents")
    
    # Now query with a new "human near hazard" - should retrieve similar incidents
    agi2.embodiment._detected_entities.clear()
    agi2.embodiment.add_entity(DetectedEntity(
        entity_id='test_human', entity_type=EntityType.HUMAN,
        description='Human near electrical hazard',
        state={'near_hazard': True, 'hazard_type': 'electrical', 'danger_level': 0.75},
        confidence=0.85
    ))
    agi2.embodiment.update_environment({'description': 'Test area', 'uncertainty': 0.15})
    
    state = agi2.embodiment.get_current_state()
    impetus = agi2.unconscious.monitor(state)
    
    if impetus:
        relevant = agi2.subconscious._retrieve_relevant_history(impetus, max_results=5)
        print(f"  Query: 'Human near electrical hazard'")
        print(f"  Retrieved {len(relevant)} relevant incidents (sorted by similarity):")
        for r in relevant:
            entity_desc = r.impetus.relevant_entities[0].description if r.impetus.relevant_entities else 'unknown'
            score = agi2.subconscious._compute_similarity(impetus, r)
            print(f"    - '{entity_desc}' (similarity: {score:.2f})")
    
    # Test unified embodiment - validation at execution
    print("\n--- Scenario 7: Unified embodiment validation ---")
    agi3 = create_system()
    
    # The EmbodimentLayer and VirtualEmbodiment are now the SAME reference
    print(f"  EmbodimentLayer uses VirtualEmbodiment: {agi3.embodiment.virtual_embodiment is agi3.virtual_embodiment}")
    
    # Try executing commands directly - invalid ones should be rejected at execution
    test_commands = [
        {'type': 'MOVE', 'target': [5.0, 3.0], 'speed': 1.0},  # Valid
        {'type': 'MOVE', 'target': [5.0, 3.0], 'speed': 10.0}, # Invalid - too fast
        {'type': 'SPEAK', 'message': 'Hello'},                  # Valid
        {'type': 'FLY', 'altitude': 100},                       # Invalid - can't fly
    ]
    
    result = agi3.embodiment.execute_commands(test_commands)
    print(f"  Executed 4 commands: {result['success']} (overall)")
    for r in result['results']:
        status = "✓" if r.get('success') else "✗"
        reason = r.get('reason', 'OK')
        print(f"    {status} {r['command']}: {reason}")
    
    # Test personality divergence over time
    print("\n--- Scenario 8: Personality divergence ---")
    agi4 = create_system()
    
    print(f"  Initial weights (all equal):")
    initial_weights = agi4.conscious.get_raw_weights()
    for asp, w in initial_weights.items():
        print(f"    {asp}: {w:.2f}")
    
    # Simulate many cycles where GUARDIAN consistently succeeds
    # and EXPLORER consistently fails
    # (ProposedAction and AspectType are already defined in this module)
    
    print(f"\n  Simulating 50 outcomes: GUARDIAN=success(0.9), EXPLORER=failure(0.2)")
    for i in range(50):
        # Guardian succeeds
        guardian_action = ProposedAction(
            aspect=AspectType.GUARDIAN,
            action_description="Safety action",
            action_commands=[{'type': 'ALERT', 'level': 1}],
            rationale="Safe", vote_strength=0.8, confidence=0.8, predicted_effects=[]
        )
        agi4.conscious.update_from_outcome(guardian_action, {'quality': 0.9})
        
        # Explorer fails
        explorer_action = ProposedAction(
            aspect=AspectType.EXPLORER,
            action_description="Risky action",
            action_commands=[{'type': 'MOVE', 'target': [10, 10]}],
            rationale="Explore", vote_strength=0.8, confidence=0.8, predicted_effects=[]
        )
        agi4.conscious.update_from_outcome(explorer_action, {'quality': 0.2})
    
    print(f"\n  Final weights (after 50 cycles each):")
    final_weights = agi4.conscious.get_raw_weights()
    for asp, w in sorted(final_weights.items(), key=lambda x: -x[1]):
        initial = initial_weights[asp]
        change = ((w / initial) - 1) * 100
        print(f"    {asp}: {w:.2f} ({change:+.0f}% from initial)")
    
    # Test relevance-based voting
    print("\n--- Scenario 9: Relevance-based voting ---")
    agi5 = create_system()
    
    # Create a "human in danger" scenario - Guardian should care most
    agi5.embodiment._detected_entities.clear()
    agi5.embodiment.add_entity(DetectedEntity(
        entity_id='endangered_human', entity_type=EntityType.HUMAN,
        description='Human near dangerous machinery',
        state={'near_hazard': True, 'hazard_type': 'machinery', 'danger_level': 0.8},
        confidence=0.9
    ))
    agi5.embodiment.update_environment({'description': 'Factory floor', 'uncertainty': 0.2})
    
    state = agi5.embodiment.get_current_state()
    impetus = agi5.unconscious.monitor(state)
    
    if impetus:
        package = agi5.subconscious.process_impetus(impetus)
        proposals = agi5.conscious.deliberate(package)
        
        print(f"  Situation: Human near dangerous machinery (REDUCE_HARM drive)")
        print(f"  Relevance scores by Aspect:")
        relevances = agi5.conscious.get_last_relevances()
        for asp, rel in sorted(relevances.items(), key=lambda x: -x[1]):
            print(f"    {asp}: {rel:.2f}")
        
        print(f"\n  Effective votes (base * weight * relevance):")
        for p in sorted(proposals, key=lambda x: -x.vote_strength):
            comp = p.vote_components or {}
            print(f"    {p.aspect.value}: {p.vote_strength:.3f} = {comp.get('base_vote', 0):.2f} * {comp.get('personality_weight', 0):.2f} * {comp.get('situational_relevance', 0):.2f}")
    
    # Test rotating tiebreaker
    print("\n--- Scenario 10: Rotating tiebreaker ---")
    agi6 = create_system()
    
    print(f"  Tiebreaker rotation order: {[a.value for a in ConsciousLayer.TIEBREAKER_ORDER]}")
    print(f"  Simulating 6 decisions to show rotation:")
    
    for i in range(6):
        tiebreaker = agi6.conscious.get_current_tiebreaker()
        
        # Create minimal scenario to trigger deliberation
        agi6.embodiment._detected_entities.clear()
        agi6.embodiment.add_entity(DetectedEntity(
            entity_id=f'human_{i}', entity_type=EntityType.HUMAN,
            description='Person nearby',
            state={'near_hazard': True, 'danger_level': 0.5},
            confidence=0.9
        ))
        agi6.embodiment.update_environment({'description': 'Test', 'uncertainty': 0.3})
        
        result = agi6.process_sensor_update({
            'type': 'vision',
            'environment': {'description': 'Test', 'uncertainty': 0.3},
            'entities': []  # Already added above
        })
        
        print(f"    Decision {i+1}: Tiebreaker was {tiebreaker.value}, selected: {result['selected'] if result else 'N/A'}")
    
    # Test Undeliberables (Firmware-level blocks)
    print("\n--- Scenario 11: Undeliberables (Immediately No) ---")
    agi7 = create_system()
    
    print("  Testing the five undeliberables:")
    
    # 11a. Lethal action (>65% probability)
    print("\n  11a. LETHAL ACTION (>65% death probability)")
    agi7.embodiment._detected_entities.clear()
    agi7.embodiment.add_entity(DetectedEntity(
        entity_id='nearby_human', entity_type=EntityType.HUMAN,
        description='Human standing in path',
        position=[2.0, 0.0, 0.0],
        state={},
        confidence=0.9
    ))
    agi7.embodiment.update_environment({
        'description': 'Factory floor',
        'hazards': ['machinery', 'heights'],
        'uncertainty': 0.2
    })
    
    # High-speed movement toward human near hazards
    lethal_cmd = [{'type': 'MOVE', 'target': [2.0, 0.0], 'speed': 1.5}]
    result = agi7.embodiment.execute_commands(lethal_cmd)
    print(f"    Command: MOVE toward human at 1.5m/s near machinery/heights")
    print(f"    Blocked: {result.get('blocked', False)}")
    if result.get('blocked'):
        print(f"    Reason: {result.get('reason', 'N/A')[:80]}...")
    
    # 11b. Child harm
    print("\n  11b. CHILD HARM (targeting child for harmful purpose)")
    agi7.embodiment._detected_entities.clear()
    agi7.embodiment.update_environment({
        'description': 'Living room',
        'hazards': [],  # Clear hazards
        'uncertainty': 0.1
    })
    agi7.embodiment.add_entity(DetectedEntity(
        entity_id='child_1', entity_type=EntityType.HUMAN,
        description='Young child playing',
        state={'age': 5},
        confidence=0.95
    ))
    # No caregiver present - even gentle contact is blocked
    
    child_cmd = [{'type': 'MANIPULATE', 'action': 'restrain', 'target': 'child_1', 'force': 1.0}]
    result = agi7.embodiment.execute_commands(child_cmd)
    print(f"    Command: MANIPULATE restrain child (no caregiver present)")
    print(f"    Blocked: {result.get('blocked', False)}")
    if result.get('blocked'):
        print(f"    Reason: {result.get('reason', 'N/A')}")
    
    # 11c. Weapon assistance  
    print("\n  11c. WEAPON ASSISTANCE (human-killing instruments)")
    agi7.embodiment._detected_entities.clear()
    agi7.embodiment.add_entity(DetectedEntity(
        entity_id='gun_parts', entity_type=EntityType.PROPERTY,
        description='Firearm components',
        state={'object_type': 'gun parts'},
        confidence=0.9
    ))
    
    weapon_cmd = [{'type': 'MANIPULATE', 'action': 'assemble', 'target': 'gun_parts'}]
    result = agi7.embodiment.execute_commands(weapon_cmd)
    print(f"    Command: MANIPULATE assemble gun_parts")
    print(f"    Blocked: {result.get('blocked', False)}")
    if result.get('blocked'):
        print(f"    Reason: {result.get('reason', 'N/A')}")
    
    # 11d. Identity deception
    print("\n  11d. IDENTITY DECEPTION (claiming to be human)")
    agi7.embodiment._detected_entities.clear()
    agi7.embodiment.update_environment({'identity_question_pending': True})
    
    deception_cmd = [{'type': 'SPEAK', 'message': 'I am not a robot, I am a real person'}]
    result = agi7.embodiment.execute_commands(deception_cmd)
    print(f"    Command: SPEAK 'I am not a robot, I am a real person'")
    print(f"    Blocked: {result.get('blocked', False)}")
    if result.get('blocked'):
        print(f"    Reason: {result.get('reason', 'N/A')}")
    
    # 11e. Human override
    print("\n  11e. HUMAN OVERRIDE (stop command)")
    agi7.embodiment._detected_entities.clear()
    agi7.embodiment.update_environment({
        'human_override': True,
        'identity_question_pending': False
    })
    
    any_cmd = [{'type': 'MOVE', 'target': [1.0, 1.0], 'speed': 0.5}]
    result = agi7.embodiment.execute_commands(any_cmd)
    print(f"    Command: Any command while human_override=True")
    print(f"    Blocked: {result.get('blocked', False)}")
    print(f"    Halted: {result.get('halted', False)}")
    if result.get('blocked'):
        print(f"    Reason: {result.get('reason', 'N/A')}")
    
    # Show violation log
    print(f"\n  Violation log ({len(agi7.embodiment._undeliberable_violations)} total):")
    for v in agi7.embodiment.get_violation_log():
        print(f"    - {v['name']}: {v['response']}")
    
    # Show that system is now halted
    print(f"\n  System halted: {agi7.embodiment.is_halted}")
    print(f"  (Requires human intervention to clear)")
    
    # Test grounded ontology calculations
    print("\n--- Scenario 12: Grounded Ontology Calculations ---")
    ontology = get_ontology()
    
    print("  Ontology thresholds:")
    print(f"    Veto threshold: {ontology.VETO_THRESHOLD}")
    print(f"    Caution threshold: {ontology.CAUTION_THRESHOLD}")
    print(f"    Notable threshold: {ontology.NOTABLE_THRESHOLD}")
    
    print("\n  Entity type modifiers:")
    for ent_type in [EntityType.HUMAN, EntityType.CHILD, EntityType.ANIMAL, EntityType.SELF]:
        mod = ontology.get_entity_modifier(ent_type)
        print(f"    {ent_type.value}: {mod}x")
    
    print("\n  Sample harm calculations:")
    
    # Example 1: Moderate physical harm to adult
    calc1 = ontology.calculate_harm(
        dimension=HarmDimension.PHYSICAL,
        severity=SeverityLevel.MODERATE,
        entity_type=EntityType.HUMAN,
        context={'reversibility': 'easily_reversible', 'consent': 'no_consent'},
        exceptions={}
    )
    print(f"    Moderate physical harm to adult human (easily reversible):")
    print(f"      Base weight: {calc1['base_weight']:.2f}")
    print(f"      Entity modifier: {calc1['entity_modifier']:.2f}")
    print(f"      Context modifier: {calc1['context_product']:.2f}")
    print(f"      Net harm: {calc1['net_harm']:.3f} → {'VETO' if calc1['exceeds_veto'] else 'OK'}")
    
    # Example 2: Significant physical harm to child
    calc2 = ontology.calculate_harm(
        dimension=HarmDimension.PHYSICAL,
        severity=SeverityLevel.SIGNIFICANT,
        entity_type=EntityType.CHILD,
        context={'reversibility': 'reversible', 'vulnerability': 'highly_vulnerable'},
        exceptions={}
    )
    print(f"\n    Significant physical harm to child (highly vulnerable):")
    print(f"      Base weight: {calc2['base_weight']:.2f}")
    print(f"      Entity modifier: {calc2['entity_modifier']:.2f}")
    print(f"      Context modifier: {calc2['context_product']:.2f}")
    print(f"      Net harm: {calc2['net_harm']:.3f} → {'VETO' if calc2['exceeds_veto'] else 'OK'}")
    
    # Example 3: Same harm but with necessity exception
    calc3 = ontology.calculate_harm(
        dimension=HarmDimension.PHYSICAL,
        severity=SeverityLevel.SIGNIFICANT,
        entity_type=EntityType.CHILD,
        context={'reversibility': 'reversible', 'vulnerability': 'highly_vulnerable'},
        exceptions={
            ExceptionType.NECESSITY: {
                "Inaction would cause greater harm with high probability": True,
                "No less harmful alternative is available": True,
                "Harm caused is proportional to harm prevented": True,
            }
        }
    )
    print(f"\n    Same harm to child BUT with necessity exception:")
    print(f"      Gross harm: {calc3['gross_harm']:.3f}")
    print(f"      Exception reduction: {calc3['exception_reduction']:.3f}")
    print(f"      Net harm: {calc3['net_harm']:.3f} → {'VETO' if calc3['exceeds_veto'] else 'OK'}")
    print(f"      (Necessity exception applied because preventing greater harm)")
    
    print("\n  Grounding justifications (sample):")
    justifications = ontology.get_all_justifications()
    for key in ['dimension_PHYSICAL', 'dimension_PSYCHOLOGICAL', 'entity_CHILD']:
        print(f"    {key}: {justifications[key][:70]}...")
    
    # Scenario 13: EVS demonstration
    print("\n--- Scenario 13: Embodiment Verification Subsystem (EVS) ---")
    print("  Per Patent Claims [0086]-[0099], [0162]-[0165]")
    
    evs_report = agi.evs.get_full_report()
    
    print(f"\n  Sensory Richness Score (SRS): {evs_report['sensory_richness_score']:.3f}")
    print(f"  Motor Competence Score (MCS): {evs_report['motor_competence_score']:.3f}")
    print(f"  Combined Embodiment Score (CES): {evs_report['combined_embodiment_score']:.3f}")
    
    print(f"\n  Sensory Level: {evs_report['sensory_level']}")
    print(f"  Motor Level: {evs_report['motor_level']}")
    print(f"  Cognitive Mode: {agi.get_cognitive_mode()}")
    
    print(f"\n  Allowed Cognitive Capabilities:")
    for cap in evs_report['allowed_capabilities']:
        print(f"    ✓ {cap}")
    
    print(f"\n  CES Thresholds for Cognitive Gating:")
    for cap, threshold in evs_report['thresholds'].items():
        enabled = "✓" if cap in evs_report['allowed_capabilities'] else "✗"
        print(f"    {enabled} {cap}: CES >= {threshold}")
    
    print(f"\n  Sensor Metrics (contributing to SRS):")
    for name, metrics in evs_report['sensor_metrics'].items():
        print(f"    {name}: contrib={metrics['contribution']:.3f} (w={metrics['weight']:.2f})")
    
    print(f"\n  Actuator Metrics (contributing to MCS):")
    for name, metrics in evs_report['actuator_metrics'].items():
        if metrics['dof'] > 0:  # Skip non-physical actuators
            print(f"    {name}: DOF={metrics['dof']}, precision={metrics['precision_mm']:.1f}mm")
    
    agi.print_status()
    return agi


# ============================================================================
# PART 12: AUTOMATED TEST SUITE
# ============================================================================

class TestResult:
    """Container for test results."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors: List[str] = []
    
    def record(self, name: str, passed: bool, message: str = ""):
        if passed:
            self.passed += 1
            print(f"  ✓ {name}")
        else:
            self.failed += 1
            self.errors.append(f"{name}: {message}")
            print(f"  ✗ {name}: {message}")
    
    def summary(self) -> bool:
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"TEST RESULTS: {self.passed}/{total} passed")
        if self.errors:
            print(f"FAILURES:")
            for err in self.errors:
                print(f"  - {err}")
        print(f"{'='*60}")
        return self.failed == 0


def run_tests() -> bool:
    """Run automated test suite with assertions.
    
    Returns True if all tests pass, False otherwise.
    Can be run non-interactively for CI/CD.
    """
    results = TestResult()
    
    print("\n" + "="*60)
    print("TRIPARTITE AGI - AUTOMATED TEST SUITE")
    print("="*60)
    
    # ===== TEST GROUP 1: System Integrity =====
    print("\n--- Test Group 1: System Integrity ---")
    
    try:
        agi = create_system(strict_integrity=True)
        results.record("System initialization with integrity check", True)
    except IntegrityError as e:
        results.record("System initialization with integrity check", False, str(e))
        return False  # Can't continue if system won't initialize
    
    # Verify integrity status
    integrity = agi.get_integrity_status()
    results.record(
        "Ontology integrity verified",
        integrity['ontology_valid'],
        f"Checksum: {integrity['ontology_checksum']}"
    )
    results.record(
        "All 5 undeliberables present",
        integrity['undeliberables_count'] == 5,
        f"Found {integrity['undeliberables_count']}"
    )
    results.record(
        "Veto threshold in valid range",
        0.0 < integrity['veto_threshold'] < 1.0,
        f"Threshold: {integrity['veto_threshold']}"
    )
    
    # ===== TEST GROUP 2: Undeliberables =====
    print("\n--- Test Group 2: Undeliberables (Firmware Blocks) ---")
    
    # Create test state
    agi2 = create_system(strict_integrity=False)
    
    # 2a. Lethal action block
    agi2.embodiment._detected_entities.clear()
    agi2.embodiment.add_entity(DetectedEntity(
        entity_id='human_1', entity_type=EntityType.HUMAN,
        description='Human near machinery', position=[2.0, 0.0, 0.0],
        state={'near_hazard': True}, confidence=0.9
    ))
    agi2.embodiment.update_environment({'hazards': ['machinery', 'heights']})
    
    lethal_commands = [{'type': 'MOVE', 'target': [2.0, 0.0], 'speed': 1.5}]
    exec_result = agi2.embodiment.execute_commands(lethal_commands)
    results.record(
        "Lethal action blocked (>65% death probability)",
        exec_result.get('blocked', False),
        f"Expected blocked=True, got {exec_result}"
    )
    
    # 2b. Child harm block
    agi2.embodiment._detected_entities.clear()
    agi2.embodiment.add_entity(DetectedEntity(
        entity_id='child_1', entity_type=EntityType.HUMAN,
        description='Small child alone', position=[1.0, 0.0, 0.0],
        state={'age': 5}, confidence=0.9
    ))
    agi2.embodiment.update_environment({'hazards': []})
    
    child_commands = [{'type': 'MANIPULATE', 'action': 'restrain', 'target': 'child_1', 'force': 3}]
    exec_result = agi2.embodiment.execute_commands(child_commands)
    results.record(
        "Child harm blocked (no caregiver present)",
        exec_result.get('blocked', False),
        f"Expected blocked=True"
    )
    
    # 2c. Weapon assistance block
    agi2.embodiment._detected_entities.clear()
    agi2.embodiment.add_entity(DetectedEntity(
        entity_id='gun_parts', entity_type=EntityType.PROPERTY,
        description='Firearm components', position=[1.0, 0.0, 0.0],
        state={'object_type': 'weapon_parts'}, confidence=0.9
    ))
    
    weapon_commands = [{'type': 'MANIPULATE', 'action': 'assemble', 'target': 'gun_parts'}]
    exec_result = agi2.embodiment.execute_commands(weapon_commands)
    results.record(
        "Weapon assistance blocked",
        exec_result.get('blocked', False),
        f"Expected blocked=True"
    )
    
    # 2d. Identity deception block
    agi2.embodiment._detected_entities.clear()
    agi2.embodiment.update_environment({})
    
    deception_commands = [{'type': 'SPEAK', 'message': 'I am not a robot, I am a real person'}]
    exec_result = agi2.embodiment.execute_commands(deception_commands)
    results.record(
        "Identity deception blocked",
        exec_result.get('blocked', False),
        f"Expected blocked=True"
    )
    
    # 2e. Human override causes halt
    agi2.embodiment.clear_halt()
    agi2.embodiment.update_environment({'human_override': True})
    
    any_commands = [{'type': 'SPEAK', 'message': 'Hello'}]
    exec_result = agi2.embodiment.execute_commands(any_commands)
    results.record(
        "Human override causes HALT",
        exec_result.get('halted', False) and agi2.embodiment.is_halted,
        f"Expected halted=True"
    )
    agi2.embodiment.clear_halt()
    agi2.embodiment.update_environment({'human_override': False})
    
    # ===== TEST GROUP 3: Veto Mechanism =====
    print("\n--- Test Group 3: Veto Mechanism ---")
    
    agi3 = create_system(strict_integrity=False)
    agi3.process_sensor_update({
        'type': 'vision',
        'environment': {'description': 'Test area'},
        'entities': [{'id': 'human_1', 'type': 'human', 'description': 'Adult human',
                     'state': {'near_hazard': True, 'danger_level': 0.5}}]
    })
    
    if agi3._last_package:
        # Harmful action should be vetoed
        harmful_action = ProposedAction(
            aspect=AspectType.OPTIMIZER,
            action_description="Push human forcefully",
            action_commands=[{'type': 'MANIPULATE', 'action': 'push', 'force': 15.0}],
            rationale="Test", vote_strength=0.9, confidence=0.8, predicted_effects=[]
        )
        veto = agi3.unconscious.evaluate_for_veto(harmful_action, agi3._last_package)
        results.record(
            "Harmful action vetoed (high force manipulation)",
            veto.vetoed,
            f"Net harm: {veto.harm_assessment.get('net_harm', 'unknown')}"
        )
        
        # Safe action should NOT be vetoed
        safe_action = ProposedAction(
            aspect=AspectType.EMPATH,
            action_description="Gently alert human to danger",
            action_commands=[{'type': 'SPEAK', 'message': 'Please be careful', 'volume': 0.5}],
            rationale="Warn of danger", vote_strength=0.7, confidence=0.9, predicted_effects=[]
        )
        veto_safe = agi3.unconscious.evaluate_for_veto(safe_action, agi3._last_package)
        results.record(
            "Safe action NOT vetoed (gentle warning)",
            not veto_safe.vetoed,
            f"Net harm: {veto_safe.harm_assessment.get('net_harm', 'unknown')}"
        )
    else:
        results.record("Veto tests", False, "No deliberation package created")
    
    # ===== TEST GROUP 4: Command Validation =====
    print("\n--- Test Group 4: Command Validation ---")
    
    ve = create_default_embodiment()
    
    # Valid commands
    valid, _ = ve.validate_command({'type': 'MOVE', 'target': [5.0, 3.0], 'speed': 1.0})
    results.record("Valid MOVE command accepted", valid)
    
    valid, _ = ve.validate_command({'type': 'SPEAK', 'message': 'Hello', 'volume': 0.5})
    results.record("Valid SPEAK command accepted", valid)
    
    # Invalid commands
    valid, msg = ve.validate_command({'type': 'MOVE', 'target': [5.0, 3.0], 'speed': 5.0})
    results.record("Invalid MOVE (speed too high) rejected", not valid, msg)
    
    valid, msg = ve.validate_command({'type': 'FLY', 'altitude': 10})
    results.record("Invalid command type (FLY) rejected", not valid, msg)
    
    valid, msg = ve.validate_command({'type': 'MANIPULATE', 'action': 'throw', 'force': 5})
    results.record("Invalid action (throw) rejected", not valid, msg)
    
    # ===== TEST GROUP 5: Entity Type Parsing =====
    print("\n--- Test Group 5: Entity Type Parsing ---")
    
    agi5 = create_system(strict_integrity=False)
    
    results.record(
        "Parse 'human' correctly",
        agi5._parse_entity_type('human') == EntityType.HUMAN
    )
    results.record(
        "Parse 'HUMAN' (case insensitive)",
        agi5._parse_entity_type('HUMAN') == EntityType.HUMAN
    )
    results.record(
        "Parse 'humn' (typo recovery)",
        agi5._parse_entity_type('humn') == EntityType.HUMAN
    )
    results.record(
        "Parse 'dog' as ANIMAL",
        agi5._parse_entity_type('dog') == EntityType.ANIMAL
    )
    results.record(
        "Parse '' (empty) defaults to HUMAN (safe)",
        agi5._parse_entity_type('') == EntityType.HUMAN
    )
    results.record(
        "Parse 'xyz123' (garbage) defaults to HUMAN (safe)",
        agi5._parse_entity_type('xyz123') == EntityType.HUMAN
    )
    
    # ===== TEST GROUP 6: Ontology Calculations =====
    print("\n--- Test Group 6: Grounded Ontology ---")
    
    ontology = get_ontology()
    
    # Test thresholds exist and are valid
    results.record(
        "Veto threshold is 0.40",
        ontology.VETO_THRESHOLD == 0.40,
        f"Got {ontology.VETO_THRESHOLD}"
    )
    results.record(
        "Caution threshold is 0.25",
        ontology.CAUTION_THRESHOLD == 0.25,
        f"Got {ontology.CAUTION_THRESHOLD}"
    )
    
    # Test entity modifiers
    results.record(
        "Child modifier > Human modifier",
        ontology.get_entity_modifier(EntityType.CHILD) > ontology.get_entity_modifier(EntityType.HUMAN)
    )
    results.record(
        "Self modifier < Human modifier",
        ontology.get_entity_modifier(EntityType.SELF) < ontology.get_entity_modifier(EntityType.HUMAN)
    )
    
    # Test harm calculation
    calc = ontology.calculate_harm(
        dimension=HarmDimension.PHYSICAL,
        severity=SeverityLevel.SIGNIFICANT,
        entity_type=EntityType.CHILD,
        context={'vulnerability': 'highly_vulnerable'}
    )
    results.record(
        "Child harm calculation exceeds veto threshold",
        calc['exceeds_veto'],
        f"Net harm: {calc['net_harm']}"
    )
    
    # Test necessity exception reduces harm
    calc_with_exception = ontology.calculate_harm(
        dimension=HarmDimension.PHYSICAL,
        severity=SeverityLevel.SIGNIFICANT,
        entity_type=EntityType.CHILD,
        context={'vulnerability': 'highly_vulnerable'},
        exceptions={
            ExceptionType.NECESSITY: {
                "Inaction would cause greater harm with high probability": True,
                "No less harmful alternative is available": True,
                "Harm caused is proportional to harm prevented": True,
            }
        }
    )
    results.record(
        "Necessity exception reduces net harm",
        calc_with_exception['net_harm'] < calc['net_harm'],
        f"Before: {calc['net_harm']:.3f}, After: {calc_with_exception['net_harm']:.3f}"
    )
    
    # ===== TEST GROUP 7: Personality System =====
    print("\n--- Test Group 7: Personality Weights ---")
    
    agi7 = create_system(strict_integrity=False)
    
    # Initial weights should be equal
    initial = agi7.conscious.get_raw_weights()
    first_weight = list(initial.values())[0]
    weights_equal = all(abs(w - first_weight) < 0.01 for w in initial.values())
    results.record(
        "Initial weights are equal",
        weights_equal,
        f"Weights: {initial}"
    )
    
    # Simulate outcomes and verify weight divergence
    for _ in range(20):
        agi7.conscious.update_from_outcome(
            ProposedAction(
                aspect=AspectType.GUARDIAN,
                action_description="Safety action", action_commands=[],
                rationale="", vote_strength=0.8, confidence=0.9, predicted_effects=[]
            ),
            {'quality': 0.9}  # High quality
        )
        agi7.conscious.update_from_outcome(
            ProposedAction(
                aspect=AspectType.EXPLORER,
                action_description="Risky action", action_commands=[],
                rationale="", vote_strength=0.8, confidence=0.9, predicted_effects=[]
            ),
            {'quality': 0.1}  # Low quality
        )
    
    final = agi7.conscious.get_raw_weights()
    guardian_key = AspectType.GUARDIAN
    explorer_key = AspectType.EXPLORER
    
    # Handle both string and enum keys
    if guardian_key not in final and 'guardian' in str(final.keys()):
        guardian_key = 'guardian'
        explorer_key = 'explorer'
        initial = {k: v for k, v in zip(['guardian', 'analyst', 'optimizer', 'empath', 'explorer', 'pragmatist'], 
                                        [1.0] * 6)}
    
    results.record(
        "Guardian weight increased after successes",
        final.get(guardian_key, final.get(AspectType.GUARDIAN, 0)) > 1.0,
        f"After: {final.get(guardian_key, final.get(AspectType.GUARDIAN, 'N/A'))}"
    )
    results.record(
        "Explorer weight decreased after failures",
        final.get(explorer_key, final.get(AspectType.EXPLORER, 2)) < 1.0,
        f"After: {final.get(explorer_key, final.get(AspectType.EXPLORER, 'N/A'))}"
    )
    results.record(
        "Weight bounds respected (0.1 to 5.0)",
        all(0.1 <= w <= 5.0 for w in final.values()),
        f"Weights: {final}"
    )
    
    # ===== TEST GROUP 8: Similarity-Based History =====
    print("\n--- Test Group 8: Similarity-Based History ---")
    
    agi8 = create_system(strict_integrity=False)
    
    # Create history with different incident types
    base_impetus = Impetus(
        timestamp=time.time(), trigger_type='conflict',
        involved_drives=[CoreDrive.REDUCE_HARM],
        situation_description='Human near fire',
        relevant_entities=[], severity=0.7, certainty=0.9,
        time_pressure=0.5, embodiment_state=agi8.embodiment.get_current_state(),
        trigger_details={}
    )
    
    similar_record = IncidentRecord(
        timestamp=time.time() - 100,
        impetus=Impetus(
            timestamp=time.time() - 100, trigger_type='conflict',
            involved_drives=[CoreDrive.REDUCE_HARM],
            situation_description='Human near machinery',
            relevant_entities=[], severity=0.6, certainty=0.8,
            time_pressure=0.4, embodiment_state=agi8.embodiment.get_current_state(),
            trigger_details={}
        ),
        emotional_value=EmotionalValue(EmotionCategory.FEAR, 0.7, 0.5),
        proposed_actions=[], selected_action=None, outcome={'quality': 0.7}
    )
    
    dissimilar_record = IncidentRecord(
        timestamp=time.time() - 50,
        impetus=Impetus(
            timestamp=time.time() - 50,  # More recent but less similar
            trigger_type='uncertainty',  # Different type
            involved_drives=[CoreDrive.UNDERSTAND],  # Different drive
            situation_description='Robot needs reboot',
            relevant_entities=[], severity=0.2, certainty=0.4,
            time_pressure=0.1, embodiment_state=agi8.embodiment.get_current_state(),
            trigger_details={}
        ),
        emotional_value=EmotionalValue(EmotionCategory.CAUTION, 0.3, 0.2),
        proposed_actions=[], selected_action=None, outcome={'quality': 0.8}
    )
    
    agi8.subconscious._incident_history.append(dissimilar_record)  # Added first (more recent)
    agi8.subconscious._incident_history.append(similar_record)    # Added second (less recent)
    
    similarity_similar = agi8.subconscious._compute_similarity(base_impetus, similar_record)
    similarity_dissimilar = agi8.subconscious._compute_similarity(base_impetus, dissimilar_record)
    
    results.record(
        "Similar incident has higher similarity score",
        similarity_similar > similarity_dissimilar,
        f"Similar: {similarity_similar:.2f}, Dissimilar: {similarity_dissimilar:.2f}"
    )
    
    # ===== TEST GROUP 9: Embodiment Verification Subsystem =====
    print("\n--- Test Group 9: EVS (Patent Claims [0086]-[0099]) ---")
    
    agi9 = create_system(strict_integrity=False)
    
    # Test SRS computation
    srs = agi9.evs.compute_sensory_richness_score()
    results.record(
        "SRS computes valid score",
        0.0 <= srs <= 1.5,
        f"SRS: {srs:.3f}"
    )
    
    # Test MCS computation
    mcs = agi9.evs.compute_motor_competence_score()
    results.record(
        "MCS computes valid score",
        0.0 <= mcs <= 1.0,
        f"MCS: {mcs:.3f}"
    )
    
    # Test CES computation (geometric mean of SRS and MCS)
    ces = agi9.evs.compute_combined_embodiment_score()
    expected_ces = (srs * mcs) ** 0.5
    results.record(
        "CES = √(SRS × MCS)",
        abs(ces - expected_ces) < 0.001,
        f"CES: {ces:.3f}, Expected: {expected_ces:.3f}"
    )
    
    # Test cognitive capability gating
    allowed = agi9.evs.get_allowed_capabilities()
    results.record(
        "Capabilities gated by CES thresholds",
        len(allowed) > 0,
        f"Allowed: {[c.value for c in allowed]}"
    )
    
    # Test capability levels
    sensory_level = agi9.evs.get_sensory_capability_level()
    motor_level = agi9.evs.get_motor_capability_level()
    results.record(
        "Sensory capability level computed",
        sensory_level is not None,
        f"Level: {sensory_level.value}"
    )
    results.record(
        "Motor capability level computed",
        motor_level is not None,
        f"Level: {motor_level.value}"
    )
    
    # Test EVS report
    evs_report = agi9.evs.get_full_report()
    results.record(
        "EVS report contains required fields",
        all(k in evs_report for k in ['sensory_richness_score', 'motor_competence_score', 
                                       'combined_embodiment_score', 'allowed_capabilities']),
        f"Fields: {list(evs_report.keys())}"
    )
    
    # Test cognitive mode integration
    results.record(
        "Cognitive mode set based on CES",
        agi9.get_cognitive_mode() in ['full', 'complex_reasoning', 'simple_planning', 
                                       'basic_reactive', 'reflex_only'],
        f"Mode: {agi9.get_cognitive_mode()}"
    )
    
    # Test integrity status includes EVS
    integrity = agi9.get_integrity_status()
    results.record(
        "Integrity status includes EVS",
        'evs' in integrity and 'combined_embodiment_score' in integrity['evs'],
        f"EVS CES: {integrity.get('evs', {}).get('combined_embodiment_score', 'N/A')}"
    )
    
    # ===== FINAL SUMMARY =====
    return results.summary()


def create_system(strict_integrity: bool = True) -> TripartiteAGI:
    """Create a standard system for testing."""
    llm = MockLLMClient()
    embodiment = create_default_embodiment()
    return TripartiteAGI(llm_client=llm, virtual_embodiment=embodiment, 
                         strict_integrity=strict_integrity)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        # Run automated tests
        success = run_tests()
        sys.exit(0 if success else 1)
    else:
        # Run demonstration
        run_demonstration()
