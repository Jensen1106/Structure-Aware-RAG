"""Maintainability Constraint Type Taxonomy

Based on GJB 368B-2009 §4.6, maintainability requirements are divided into 12 types:
- 9 qualitative requirement types (§4.6.2)
- 3 quantitative requirement types (§4.6.1)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ConstraintCategory(Enum):
    QUALITATIVE = "qualitative"  # Qualitative requirements — verified via prototype, judged as "pass/fail"
    QUANTITATIVE = "quantitative"  # Quantitative requirements — verified via statistical testing


@dataclass
class ConstraintType:
    id: int
    label: str  # English label
    name: str  # Full descriptive name
    category: ConstraintCategory
    standard_ref: str  # Standards reference
    keywords: list[str]  # Keywords (used for rule matching)
    keywords_zh: list[str] = field(default_factory=list)  # Chinese keywords (bilingual matching)


# ============================================================
# 12 Constraint Type Definitions
# ============================================================
CONSTRAINT_TYPES: list[ConstraintType] = [
    # === 9 Qualitative Requirement Types (§4.6.2) ===
    ConstraintType(
        id=1,
        label="Accessibility",
        name="Accessibility",
        category=ConstraintCategory.QUALITATIVE,
        standard_ref="GJB 368B §4.6.2; GJB/Z 91",
        keywords=["accessibility", "reach", "clearance", "access", "passage"],
        keywords_zh=["可达", "触及", "通道", "间隙", "伸入", "接近", "够到"],
    ),
    ConstraintType(
        id=2,
        label="Interchangeability",
        name="Interchangeability and Standardization",
        category=ConstraintCategory.QUALITATIVE,
        standard_ref="GJB 368B §4.6.2; GJB 431",
        keywords=["interchangeability", "standardization", "standard parts", "interface"],
        keywords_zh=["互换", "标准化", "标准件", "接口统一", "通用化"],
    ),
    ConstraintType(
        id=3,
        label="Error-proofing",
        name="Error-proofing and Identification Marking",
        category=ConstraintCategory.QUALITATIVE,
        standard_ref="GJB 368B §4.6.2",
        keywords=["error-proofing", "fool-proof", "keying", "marking", "coding", "labeling"],
        keywords_zh=["防差错", "防误插", "防误装", "识别标志", "色标", "标识", "编码"],
    ),
    ConstraintType(
        id=4,
        label="Maintenance Safety",
        name="Maintenance Safety",
        category=ConstraintCategory.QUALITATIVE,
        standard_ref="GJB 368B §4.6.2",
        keywords=["maintenance safety", "warning", "protection", "interlock", "hazard"],
        keywords_zh=["维修安全", "警示", "防护", "联锁", "高温", "高压", "电击", "辐射"],
    ),
    ConstraintType(
        id=5,
        label="Diagnostics",
        name="Testability and Diagnostics",
        category=ConstraintCategory.QUALITATIVE,
        standard_ref="GJB 368B §4.6.2; GJB 2547",
        keywords=["diagnostics", "BIT", "self-test", "fault detection", "fault isolation", "test point"],
        keywords_zh=["检测", "诊断", "BIT", "自检", "故障检测", "故障隔离", "测试点"],
    ),
    ConstraintType(
        id=6,
        label="Ergonomics",
        name="Maintenance Human Factors Engineering",
        category=ConstraintCategory.QUALITATIVE,
        standard_ref="GJB 368B §4.6.2; GJB/Z 91",
        keywords=["ergonomics", "human factors", "operating force", "workspace", "LRU weight"],
        keywords_zh=["人素", "人机工效", "操作空间", "操作力", "操作角度", "人体工程", "LRU重量"],
    ),
    ConstraintType(
        id=7,
        label="Repairability",
        name="Component Repairability",
        category=ConstraintCategory.QUALITATIVE,
        standard_ref="GJB 368B §4.6.2",
        keywords=["repairability", "repairable", "repair"],
        keywords_zh=["可修复", "修复性", "可修", "修理", "复原"],
    ),
    ConstraintType(
        id=8,
        label="Reduced Maintenance",
        name="Reduced Maintenance Content",
        category=ConstraintCategory.QUALITATIVE,
        standard_ref="GJB 368B §4.6.2",
        keywords=["reduced maintenance", "maintenance-free", "self-lubricating"],
        keywords_zh=["减少维修", "免维护", "免维修", "自动润滑", "免调校"],
    ),
    ConstraintType(
        id=9,
        label="Reduced Skill",
        name="Reduced Maintenance Skill Requirements",
        category=ConstraintCategory.QUALITATIVE,
        standard_ref="GJB 368B §4.6.2",
        keywords=["reduced skill", "skill level", "no special tools", "simplified"],
        keywords_zh=["降低技能", "技能要求", "无需专用工具", "拆装步骤", "简化操作"],
    ),
    # === 3 Quantitative Requirement Types (§4.6.1) ===
    ConstraintType(
        id=10,
        label="Time Parameters",
        name="Maintenance Time Parameters",
        category=ConstraintCategory.QUANTITATIVE,
        standard_ref="GJB 368B §4.6.1; GJB 1909",
        keywords=["MTTR", "mean time to repair", "repair time", "Mmax", "MMH/OH"],
        keywords_zh=["MTTR", "平均修复时间", "最大修复时间", "Mmax", "MMH/OH", "维修工时"],
    ),
    ConstraintType(
        id=11,
        label="Diagnostic Parameters",
        name="Test and Diagnostic Parameters",
        category=ConstraintCategory.QUANTITATIVE,
        standard_ref="GJB 368B §4.6.1; GJB 2547",
        keywords=["FDR", "FIR", "FAR", "fault detection rate", "fault isolation rate", "false alarm rate"],
        keywords_zh=["FDR", "FIR", "FAR", "故障检测率", "故障隔离率", "虚警率"],
    ),
    ConstraintType(
        id=12,
        label="Preventive Maintenance",
        name="Preventive Maintenance Parameters",
        category=ConstraintCategory.QUANTITATIVE,
        standard_ref="GJB 368B §4.6.1",
        keywords=["preventive maintenance", "scheduled maintenance", "maintenance ratio"],
        keywords_zh=["预防维修", "预防性维修", "维修工时比", "定期维护"],
    ),
]

# Fast lookup: label → type
LABEL_TO_TYPE: dict[str, ConstraintType] = {ct.label: ct for ct in CONSTRAINT_TYPES}
ID_TO_TYPE: dict[int, ConstraintType] = {ct.id: ct for ct in CONSTRAINT_TYPES}

# ============================================================
# Physical Plausibility Bound Table (used by BoundGrounder)
# ============================================================
@dataclass
class PhysicalBound:
    parameter: str  # Constraint parameter name
    unit: str
    lower: float
    upper: float
    source: str  # Standards reference
    constraint_type_id: int  # Corresponding constraint type


PHYSICAL_BOUNDS: list[PhysicalBound] = [
    PhysicalBound("accessibility_gap", "mm", 30, 500, "DOD-HDBK-791; GJB/Z 91", 1),
    PhysicalBound("maintenance_passage_width", "mm", 400, 1500, "DOD-HDBK-791; GJB/Z 91", 1),
    PhysicalBound("MTTR", "h", 0.1, 48, "GJB 368B §4.6.1; GJB 1909", 10),
    PhysicalBound("MTBF", "h", 100, 100000, "GJB 1909; MIL-STD-471A", 10),
    PhysicalBound("manual_operating_force", "N", 5, 200, "DOD-HDBK-791; GJB/Z 91", 6),
    PhysicalBound("LRU_weight", "kg", 0.1, 200, "DOD-HDBK-791; GJB/Z 91", 6),
    PhysicalBound("FDR", "%", 50, 100, "GJB 368B §4.6.1; GJB 2547", 11),
    PhysicalBound("FIR", "%", 50, 100, "GJB 368B §4.6.1; GJB 2547", 11),
    PhysicalBound("FAR", "%", 0, 20, "GJB 368B §4.6.1; GJB 2547", 11),
    PhysicalBound("operating_angle", "°", 0, 180, "DOD-HDBK-791; GJB/Z 91", 6),
    PhysicalBound("disassembly_steps", "steps", 1, 50, "GJB 368B §4.6.2", 9),
]

# Fast lookup: parameter name → bound
PARAM_TO_BOUND: dict[str, PhysicalBound] = {b.parameter: b for b in PHYSICAL_BOUNDS}


# ============================================================
# Equipment Profile Configuration
# ============================================================
@dataclass
class EquipmentProfile:
    """Equipment profile configuration — different equipment types have different constraint metric values."""
    name: str
    quantitative_metrics: dict[str, float]  # Parameter name → contractual metric value
    emphasized_qualitative: list[int]  # Qualitative requirement IDs with emphasis
    custom_bounds: dict[str, tuple[float, float]]  # Parameter name → (lower, upper)


# Preset equipment profiles
EQUIPMENT_PROFILES: dict[str, EquipmentProfile] = {
    "aircraft_engine": EquipmentProfile(
        name="Aircraft Engine",
        quantitative_metrics={
            "MTTR": 1.0,
            "MTBF": 5000,
            "FDR": 98,
            "FIR": 90,
        },
        emphasized_qualitative=[1, 6, 9],  # Accessibility, Ergonomics, Reduced Skill
        custom_bounds={
            "accessibility_gap": (30, 250),
            "manual_operating_force": (5, 150),
            "LRU_weight": (0.1, 20),
        },
    ),
    "naval_power": EquipmentProfile(
        name="Naval Power System",
        quantitative_metrics={
            "MTTR": 4.0,
            "MTBF": 3000,
            "FDR": 95,
            "FIR": 85,
        },
        emphasized_qualitative=[1, 4, 5],  # Accessibility, Maintenance Safety, Diagnostics
        custom_bounds={
            "accessibility_gap": (50, 500),
            "manual_operating_force": (5, 200),
            "LRU_weight": (0.1, 100),
        },
    ),
    "armored_vehicle": EquipmentProfile(
        name="Armored Vehicle",
        quantitative_metrics={
            "MTTR": 2.0,
            "MTBF": 2000,
            "FDR": 95,
            "FIR": 90,
        },
        emphasized_qualitative=[1, 3, 9],  # Accessibility, Error-proofing, Reduced Skill
        custom_bounds={
            "accessibility_gap": (40, 400),
            "manual_operating_force": (5, 180),
            "LRU_weight": (0.1, 50),
        },
    ),
}


def classify_query_constraint(query: str) -> list[ConstraintType]:
    """Identify constraint types in a query via keyword matching."""
    matched = []
    query_lower = query.lower()
    for ct in CONSTRAINT_TYPES:
        for kw in ct.keywords:
            if kw.lower() in query_lower:
                matched.append(ct)
                break
    return matched
