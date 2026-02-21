from __future__ import annotations


class PromptBuilder:
    """Constructs SDXL prompts for realistic age-specific portrait generation."""

    _AGE_DESCRIPTORS: list[tuple[tuple[int, int], str]] = [
        ((0, 2),   "newborn baby face, chubby cheeks, smooth delicate skin, soft features"),
        ((3, 6),   "toddler face, round chubby cheeks, innocent wide eyes, baby fat"),
        ((7, 11),  "child face, elementary school age, clear skin, youthful features"),
        ((12, 17), "teenage face, adolescent features, slight acne, developing bone structure"),
        ((18, 29), "young adult face, early twenties, smooth skin, strong jawline"),
        ((30, 44), "adult face, early thirties, subtle laugh lines, mature features"),
        ((45, 59), "middle-aged face, forties, crow's feet, distinguished features, salt-and-pepper hints"),
        ((60, 74), "senior face, sixties, graceful deep wrinkles, silver hair, wise eyes"),
        ((75, 100), "elderly face, seventies to eighties, deep wrinkles, age spots, gentle expression"),
    ]

    _BASE_POSITIVE = (
        "ultra-realistic portrait photograph, professional studio lighting, "
        "sharp focus, 8k resolution, cinematic, natural skin texture, "
        "consistent facial identity, same person, photorealistic"
    )

    _BASE_NEGATIVE = (
        "cartoon, anime, illustration, painting, drawing, digital art, "
        "deformed, blurry, bad anatomy, multiple faces, duplicate, "
        "watermark, text, logo, lowres, worst quality, bad quality, "
        "ugly, disfigured, mutated, extra limbs"
    )

    @classmethod
    def build(cls, age: int, suffix: str = "") -> tuple[str, str]:
        descriptor = ""
        for (lo, hi), desc in cls._AGE_DESCRIPTORS:
            if lo <= age <= hi:
                descriptor = desc
                break

        positive = f"{descriptor}, {age} years old, {cls._BASE_POSITIVE}"
        if suffix:
            positive = f"{positive}, {suffix}"

        return positive, cls._BASE_NEGATIVE
