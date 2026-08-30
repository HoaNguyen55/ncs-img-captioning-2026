"""Vietnamese lexical resources.

All Vietnamese word lists live here, in one file, so they can be inspected,
corrected and cited as a unit. Every other module in `rescap.vi` imports from
this one and adds no vocabulary of its own.

These lists are **hand-built and therefore incomplete**. That is a stated
limitation (`formulation/02 §9.5`), not an oversight: an unknown word falls back
to string equality, which under-counts matches rather than inventing one.

Sources: standard Vietnamese grammar references for the classifier system, plus
the colour and spatial inventories fixed in `formulation/02 §4.5-4.6`.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Classifiers (loại từ)
# ---------------------------------------------------------------------------
# Vietnamese noun phrases require a classifier: NUMERAL + CLASSIFIER + NOUN.
# `ba con chó` (three dogs), `ba cái ghế` (three chairs) differ because dogs are
# animate -- the classifier agrees with the noun's semantic class.
#
# DESIGN NOTE: the classifier is grammatical agreement, NOT a property of the
# referent. It lives in the Entity record and is deliberately kept out of
# `attributes` so it cannot inflate attribute precision (formulation/02 §4.1).

CLASSIFIERS: dict[str, str] = {
    "người": "humans, respectful",
    "con": "animals; some vehicles and tools",
    "cái": "inanimate objects, general",
    "chiếc": "vehicles, paired items, single objects (more formal than cái)",
    "bức": "pictures, walls, letters",
    "tấm": "flat sheets: photos, boards, cloth",
    "ngôi": "houses, stars, temples",
    "toà": "large buildings",
    "quả": "round objects, fruit",
    "trái": "round objects, fruit (southern variant of quả)",
    "cây": "trees; long thin objects",
    "đôi": "natural pairs",
    "chùm": "clusters, bunches",
    "cuốn": "books, volumes",
    "tờ": "sheets of paper, newspapers",
    "món": "dishes, items",
    "chú": "small animals, affectionate",
    "bộ": "sets, suits",
}

# Head noun -> the classifier it takes. Only entries we are confident about.
# An unknown noun gets CLASSIFIER_DEFAULT and is logged, never guessed silently.
NOUN_CLASSIFIER: dict[str, str] = {
    # people
    "người": "người", "đàn ông": "người", "phụ nữ": "người",
    "bé trai": "bé", "bé gái": "bé", "trẻ em": "đứa", "em bé": "em",
    "học sinh": "người", "công nhân": "người", "cô gái": "cô", "chàng trai": "chàng",
    # animals
    "chó": "con", "mèo": "con", "chim": "con", "cá": "con", "bò": "con",
    "trâu": "con", "gà": "con", "vịt": "con", "lợn": "con", "ngựa": "con",
    "voi": "con", "khỉ": "con", "rắn": "con", "bướm": "con", "ong": "con",
    # vehicles
    "xe": "chiếc", "xe đạp": "chiếc", "xe máy": "chiếc", "ô tô": "chiếc", "xe buýt": "chiếc",
    "xe tải": "chiếc", "tàu": "con", "thuyền": "con", "máy bay": "chiếc",
    "xích lô": "chiếc", "xe ba gác": "chiếc",
    # objects
    "bàn": "cái", "ghế": "cái", "cửa": "cái", "túi": "cái", "mũ": "cái",
    "nón": "cái", "nón lá": "cái", "áo": "chiếc", "quần": "chiếc", "giày": "đôi",
    "dép": "đôi", "kính": "cặp", "đồng hồ": "cái", "điện thoại": "cái",
    "ô": "cái", "dù": "cây", "bút": "cây", "dao": "con", "chổi": "cây",
    "bát": "cái", "chén": "cái", "đĩa": "cái", "ly": "cái", "cốc": "cái",
    # structures / nature
    "nhà": "ngôi", "toà nhà": "toà", "cầu": "cây", "cây": "cây",
    "hoa": "bông", "quả": "quả", "bóng": "quả", "núi": "ngọn", "sông": "con",
    "đường": "con", "chợ": "cái", "trường": "ngôi",
    # Compounds met on real KTVIC images, each added because its absence made
    # the parser take the head noun to be a different word. `cánh đồng hoa` was
    # read as `hoa` and realised `một bông hoa lớn` -- a large blossom, for a
    # field of flowers -- because `hoa` is in this table and `cánh đồng` was
    # not. Longest-match only wins over entries that exist.
    "cánh đồng": "cánh đồng", "khu rừng": "khu", "vỉa hè": "vỉa hè",
    "con đường": "con", "dãy nhà": "dãy", "mỏm đá": "mỏm", "tảng đá": "tảng",
    "bãi biển": "bãi", "dòng sông": "dòng", "bầu trời": "bầu",
    "ánh sáng": "ánh sáng", "mặt trời": "mặt trời",
    # media
    "tranh": "bức", "ảnh": "tấm", "sách": "cuốn", "báo": "tờ",
    "bức tranh": "bức", "biển hiệu": "tấm", "biển báo": "tấm",
    # boat parts -- `bánh lái` truncated to `bánh` put cakes in a caption of a
    # model boat, which is how this whole class of bug was found
    "bánh lái": "cái", "cọc neo": "cái", "buồm": "cánh", "mái chèo": "mái",
    # Met on KTVIC references and truncated without these: `xe tăng` was read
    # as `xe`, and the couple in `cặp đôi` / `đôi bạn trẻ` never reached
    # `người`, so a caption correctly saying `người` counted as invented.
    "xe tăng": "chiếc", "mô hình": "cái", "phòng": "căn", "căn phòng": "căn",
    "cặp đôi": "cặp", "đôi bạn": "đôi", "thanh niên": "người",
    "nam thanh niên": "người", "cô gái trẻ": "cô", "bạn trẻ": "người",
    "sạp hàng": "cái", "quầy hàng": "cái", "xe đẩy": "chiếc",
    # generic fallbacks -- without these, a common noun with no specific
    # entry falls through to the inference path and emits a warning
    "vật": "cái", "đồ": "cái", "con vật": "con", "đứa trẻ": "đứa",
}

CLASSIFIER_DEFAULT = "cái"

# ---------------------------------------------------------------------------
# Number / plurality
# ---------------------------------------------------------------------------
# Vietnamese has NO plural inflection. Plurality is lexical.
NUMBER_MARKERS: dict[str, str] = {
    "một": "singular",
    "những": "definite plural",
    "các": "definite plural (all of them)",
    "mấy": "small indefinite plural, informal",
    "một vài": "a few",
    "vài": "a few",
    "nhiều": "many",
    "một nhóm": "a group",
    "một đàn": "a herd/flock",
}

NUMERALS: dict[str, int] = {
    "không": 0, "một": 1, "hai": 2, "ba": 3, "bốn": 4, "năm": 5,
    "sáu": 6, "bảy": 7, "tám": 8, "chín": 9, "mười": 10,
}

# Beyond this, human counts disagree too -- realise approximately instead of
# asserting a number (formulation/02 §4.2).
EXACT_COUNT_LIMIT = 5

# ---------------------------------------------------------------------------
# Gender -- lexical in Vietnamese, not grammatical
# ---------------------------------------------------------------------------
# CRITICAL: the default is neutral. Guessing gender from hair or clothing is
# inference, not observation, and is scored as a hallucination
# (formulation/02 §4.3). In Vietnamese gender sits in the NOUN, so a wrong guess
# is a content error, not a pronoun-agreement slip.

GENDERED_NOUNS: dict[str, str] = {
    "đàn ông": "nam", "nam giới": "nam", "bé trai": "nam", "chàng trai": "nam",
    "ông": "nam", "anh": "nam", "cậu bé": "nam",
    "phụ nữ": "nu", "nữ giới": "nu", "bé gái": "nu", "cô gái": "nu",
    "bà": "nu", "chị": "nu", "cô": "nu",
}

NEUTRAL_PERSON = "người"

# Third-person pronouns. Gendered forms may only be used when gender is
# `clearly_visible` (formulation/11 §5.2).
PRONOUNS: dict[str, str] = {
    "nam": "anh ấy",
    "nu": "cô ấy",
    "khong_xac_dinh": "người này",
}
PRONOUN_PLURAL = "họ"

# ---------------------------------------------------------------------------
# Colour -- the `xanh` problem
# ---------------------------------------------------------------------------
# Vietnamese `xanh` covers BOTH blue and green. This has no English counterpart
# and is a genuine Vietnamese-only failure mode (formulation/02 §4.5).

COLORS: set[str] = {
    "đỏ", "vàng", "đen", "trắng", "nâu", "xám", "hồng", "tím", "cam",
    "be", "bạc", "vàng kim", "xanh dương", "xanh lam", "xanh lá", "xanh lục",
    "xanh nước biển", "xanh da trời", "xanh rêu",
}

# Surface form -> resolved value. `xanh` alone is deliberately absent: it is
# under-specified and must never resolve silently.
XANH_BLUE = {"xanh dương", "xanh lam", "xanh nước biển", "xanh da trời"}
XANH_GREEN = {"xanh lá", "xanh lục", "xanh lá cây", "xanh rêu"}
XANH_AMBIGUOUS = "xanh"

COLOR_MODIFIERS: set[str] = {"đậm", "nhạt", "sẫm", "tươi", "sáng", "tối"}

# ---------------------------------------------------------------------------
# Spatial relations -- CLOSED vocabulary
# ---------------------------------------------------------------------------
# An open string would make spatial scoring incomparable across systems
# (formulation/02 §4.6).

SPATIAL_RELATIONS: dict[str, str] = {
    "trên": "on / above",
    "dưới": "under / below",
    "trong": "inside",
    "ngoài": "outside",
    "bên cạnh": "beside",
    "cạnh": "beside",
    "phía trước": "in front of",
    "phía sau": "behind",
    "bên trái": "to the left of",
    "bên phải": "to the right of",
    "ở giữa": "between / in the middle of",
    "gần": "near",
    "xa": "far from",
    "trên đầu": "above the head of",
    "dọc theo": "along",
    "xung quanh": "around",
    "đối diện": "opposite",
}

# Not decidable from a 2-D bounding box -- caps the verdict at UNCERTAIN.
DEPTH_DEPENDENT = {"phía trước", "phía sau"}

# ---------------------------------------------------------------------------
# Aspect -- a realisation choice, never a verified fact
# ---------------------------------------------------------------------------
ASPECT_MARKERS: dict[str, str] = {
    "đang": "tiếp_diễn",     # progressive
    "đã": "hoàn_thành",      # completed
    "sẽ": "tương_lai",       # future -- never visually verifiable
    "vừa": "hoàn_thành",     # just now -- not visually verifiable
}

# ---------------------------------------------------------------------------
# Synonyms -> canonical form
# ---------------------------------------------------------------------------
# Regional variants (bắc/trung/nam) collapse to one canonical form; the original
# is preserved in `raw_text_vi` (formulation/02 §5.2).

SYNONYMS: dict[str, str] = {
    "xe hơi": "ô tô", "xe ôtô": "ô tô", "ôtô": "ô tô", "xe con": "ô tô",
    "xe gắn máy": "xe máy", "mô tô": "xe máy",
    "người đàn ông": "đàn ông", "nam giới": "đàn ông",
    "người phụ nữ": "phụ nữ", "nữ giới": "phụ nữ",
    "chén": "bát", "tô": "bát",
    "nón": "mũ",
    "bông": "hoa",
    "trái": "quả",
    "ly": "cốc",
    "xanh lam": "xanh dương", "xanh nước biển": "xanh dương",
    "xanh da trời": "xanh dương", "xanh lục": "xanh lá", "xanh lá cây": "xanh lá",
    "con đường": "đường", "đường phố": "đường",
    "chiếc xe đạp": "xe đạp",
}

# ---------------------------------------------------------------------------
# Category lattice -- shallow hypernymy
# ---------------------------------------------------------------------------
# Used by coreference (formulation/02 §2.3) and redundancy detection. Siblings
# are INCOMPATIBLE: an entity cannot be both `đàn ông` and `phụ nữ`.

HYPERNYMS: dict[str, str] = {
    "cặp đôi": "người", "đôi bạn": "người", "thanh niên": "người",
    "nam thanh niên": "đàn ông", "bạn trẻ": "người", "cô gái trẻ": "phụ nữ",
    "đàn ông": "người", "phụ nữ": "người", "trẻ em": "người",
    "bé trai": "đàn ông", "bé gái": "phụ nữ",
    "xe đạp": "phương tiện", "xe máy": "phương tiện", "ô tô": "phương tiện",
    "xe buýt": "phương tiện", "xe tải": "phương tiện", "thuyền": "phương tiện",
    "chó": "động vật", "mèo": "động vật", "chim": "động vật", "cá": "động vật",
    "bàn": "đồ đạc", "ghế": "đồ đạc",
    "áo": "quần áo", "quần": "quần áo", "mũ": "quần áo", "giày": "quần áo",
}

# ---------------------------------------------------------------------------
# Non-visual inference markers
# ---------------------------------------------------------------------------
# A proposition containing these can never be SUPPORTED on visual evidence
# alone (formulation/09 §5.1). Purpose, profession, emotion, identity are not
# readable off pixels.

NON_VISUAL_MARKERS: dict[str, str] = {
    "đi làm": "purpose", "đi học": "purpose", "đi chợ": "purpose",
    "chuẩn bị": "intention", "định": "intention", "muốn": "intention",
    "vui": "emotion", "buồn": "emotion", "tức giận": "emotion",
    "hạnh phúc": "emotion", "lo lắng": "emotion", "vội": "emotion",
    "bác sĩ": "profession", "giáo viên": "profession", "kỹ sư": "profession",
    "vợ chồng": "identity", "anh em": "identity", "bạn bè": "identity",
    "vừa mới": "temporal_before_after", "sắp": "temporal_before_after",
}

# Observable counterparts, for the guideline's contrast table.
OBSERVABLE_ALTERNATIVES: dict[str, str] = {
    "vui": "đang cười",
    "đi làm": "đang đi trên đường",
    "bác sĩ": "mặc áo blouse trắng",
    "vợ chồng": "đang đứng cạnh nhau",
    "đang ngủ": "mắt nhắm",
}

__all__ = [
    "CLASSIFIERS", "NOUN_CLASSIFIER", "CLASSIFIER_DEFAULT",
    "NUMBER_MARKERS", "NUMERALS", "EXACT_COUNT_LIMIT",
    "GENDERED_NOUNS", "NEUTRAL_PERSON", "PRONOUNS", "PRONOUN_PLURAL",
    "COLORS", "XANH_BLUE", "XANH_GREEN", "XANH_AMBIGUOUS", "COLOR_MODIFIERS",
    "SPATIAL_RELATIONS", "DEPTH_DEPENDENT",
    "ASPECT_MARKERS", "SYNONYMS", "HYPERNYMS",
    "NON_VISUAL_MARKERS", "OBSERVABLE_ALTERNATIVES",
]
