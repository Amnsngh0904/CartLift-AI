"""
Item Categorization Module

Classifies menu items into categories (main, side, dessert, beverage)
and infers vegetarian status using rule-based keyword matching.

Usage:
    python -m src.data_generation.item_categorizer --input data/processed/menu_items_cleaned.csv
    python -m src.data_generation.item_categorizer --input data/processed/menu_items_cleaned.csv --use-llm-fallback
"""

import argparse
import logging
import re
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Enums & Data Classes
# -----------------------------------------------------------------------------

class ItemCategory(str, Enum):
    """Item category enumeration."""
    MAIN = "main"
    SIDE = "side"
    DESSERT = "dessert"
    BEVERAGE = "beverage"
    UNKNOWN = "unknown"


class VegStatus(int, Enum):
    """Vegetarian status enumeration."""
    NON_VEG = 0
    VEG = 1
    UNKNOWN = -1


@dataclass
class CategoryResult:
    """Result of item categorization."""
    category: ItemCategory
    confidence: float
    matched_keywords: List[str]


@dataclass
class VegResult:
    """Result of vegetarian classification."""
    is_veg: VegStatus
    confidence: float
    matched_keywords: List[str]


# -----------------------------------------------------------------------------
# Keyword Dictionaries (Expanded for <5% unknown rate)
# -----------------------------------------------------------------------------

# Beverage keywords (highest priority - check first)
BEVERAGE_KEYWORDS = {
    # Soft drinks & sodas
    "coke", "coca cola", "coca-cola", "pepsi", "sprite", "fanta", "mirinda",
    "thumbs up", "thums up", "limca", "7up", "7-up", "mountain dew", "dew",
    "soda", "soft drink", "aerated", "carbonated", "fizz", "fizzy",
    "ginger ale", "tonic water", "club soda", "diet coke", "cola",
    "maaza", "frooti", "slice", "appy", "real juice", "tropicana",
    # Juices (expanded)
    "juice", "juices", "fresh juice", "orange juice", "apple juice", "mango juice",
    "pineapple juice", "watermelon juice", "mixed fruit juice", "lime juice",
    "lemon juice", "nimbu pani", "shikanji", "jaljeera", "aam panna",
    "sugarcane juice", "ganne ka juice", "mosambi juice", "sweet lime",
    "pomegranate juice", "grape juice", "carrot juice", "beetroot juice",
    "jamun juice", "kokum", "kokam", "sol kadhi", "raw mango", "keri",
    # Shakes & smoothies (expanded)
    "shake", "shakes", "milkshake", "milkshakes", "smoothie", "smoothies",
    "frappe", "frappuccino", "thick shake", "thickshake",
    "cold coffee shake", "oreo shake", "chocolate shake", "strawberry shake",
    "mango shake", "banana shake", "vanilla shake", "butterscotch shake",
    "kitkat shake", "nutella shake", "protein shake", "health shake",
    # Coffee & tea (expanded extensively)
    "coffee", "coffees", "espresso", "cappuccino", "latte", "americano", "mocha",
    "cold coffee", "iced coffee", "filter coffee", "kaapi", "kaffe",
    "macchiato", "flat white", "cortado", "affogato", "lungo", "ristretto",
    "cold brew", "nitro coffee", "irish coffee", "vietnamese coffee",
    "dalgona", "frappe coffee", "caramel coffee", "hazelnut coffee",
    "tea", "teas", "chai", "masala chai", "green tea", "iced tea", "lemon tea",
    "ginger tea", "kadak chai", "cutting chai", "irani chai", "sulaimani",
    "black tea", "herbal tea", "jasmine tea", "chamomile tea", "mint tea",
    "earl grey", "english breakfast", "darjeeling", "assam tea",
    "matcha", "bubble tea", "boba", "thai tea", "pink tea", "kashmiri chai",
    # Lassi & dairy drinks (expanded)
    "lassi", "lassis", "sweet lassi", "salted lassi", "mango lassi", "rose lassi",
    "chaas", "buttermilk", "mattha", "chhaas", "mor", "majjiga",
    "badam milk", "badam shake", "kesar milk", "haldi doodh", "turmeric latte",
    "thandai", "sardai", "rose milk", "strawberry milk",
    "hot chocolate", "drinking chocolate", "cocoa", "hot cocoa",
    # Mocktails & coolers
    "mojito", "mocktail", "mocktails", "cocktail", "virgin mojito", "blue lagoon",
    "lemonade", "lemonades", "nimbu soda", "lime soda", "fresh lime",
    "cooler", "coolers", "refresher", "squash", "sharbat", "sherbet",
    "iced", "chilled", "frozen", "slush", "slushie", "icee",
    "sangria", "punch", "fruit punch", "virgin sangria",
    # Water & energy drinks
    "water", "mineral water", "sparkling water", "packaged water",
    "energy drink", "red bull", "monster", "gatorade", "electrolyte",
    # Regional & traditional
    "jigarthanda", "paneer soda", "nannari", "khus", "rose sherbet",
    "sattu drink", "bael juice", "wood apple", "falsa", "litchi juice",
}

# Dessert keywords (expanded extensively)
DESSERT_KEYWORDS = {
    # Indian sweets (mithai) - extensive
    "gulab jamun", "gulabjamun", "gulabjanun", "rasgulla", "rasogolla", "rosogolla",
    "rasmalai", "ras malai", "rasamalai", "jalebi", "jilebi", "imarti", "amriti",
    "rabri", "rabdi", "malpua", "malpoa", "kheer", "payasam", "payesh",
    "halwa", "halva", "gajar halwa", "gajar ka halwa", "moong dal halwa",
    "besan halwa", "suji halwa", "sooji halwa", "badam halwa", "atta halwa",
    "ladoo", "laddu", "laddoo", "motichoor ladoo", "besan ladoo", "boondi ladoo",
    "barfi", "burfi", "barfee", "kaju barfi", "kaju katli", "pista barfi",
    "peda", "pedha", "mathura peda", "dharwad peda", "sandesh", "sandeh",
    "cham cham", "chamcham", "kalakand", "mysore pak", "soan papdi", "son papdi",
    "petha", "agra petha", "patisa", "gujiya", "modak", "puran poli",
    "phirni", "firni", "seviyan", "sevai", "vermicelli kheer", "semiya payasam",
    "basundi", "shrikhand", "amrakhand", "mishti doi", "bhapa doi",
    "chenna", "chhena", "ras bali", "chum chum", "langcha", "pantua",
    "kala jamun", "balushahi", "ghevar", "mawa", "khoya", "khoa",
    "sohan halwa", "karachi halwa", "bombay halwa", "double ka meetha",
    "qubani ka meetha", "sheer khurma", "seviyan kheer",
    # Kulfi & frozen Indian
    "kulfi", "kulfis", "malai kulfi", "pista kulfi", "mango kulfi", "kesar kulfi",
    "badam kulfi", "paan kulfi", "sitaphal kulfi", "falooda", "faluda",
    # Western desserts - ice cream
    "ice cream", "icecream", "ice-cream", "gelato", "sorbet", "frozen yogurt",
    "froyo", "sundae", "sundaes", "banana split", "parfait", "cone",
    "scoop", "scoops", "tutti frutti", "butterscotch ice cream",
    # Baked desserts (expanded)
    "brownie", "brownies", "chocolate brownie", "walnut brownie", "fudge brownie",
    "cake", "cakes", "pastry", "pastries", "cupcake", "cupcakes",
    "cheesecake", "red velvet", "black forest", "white forest",
    "chocolate cake", "vanilla cake", "fruit cake", "carrot cake",
    "truffle", "truffles", "chocolate truffle", "truffle cake",
    "mousse", "chocolate mousse", "mango mousse", "strawberry mousse",
    "tiramisu", "panna cotta", "creme brulee", "souffle",
    "pudding", "puddings", "caramel pudding", "bread pudding", "custard",
    "pie", "pies", "apple pie", "tart", "tarts", "fruit tart", "egg tart",
    "waffle", "waffles", "pancake", "pancakes", "crepe", "crepes",
    "churros", "churro", "donuts", "doughnuts", "donut", "doughnut",
    "cookie", "cookies", "biscuit", "biscuits", "macaron", "macarons",
    "eclair", "profiterole", "danish", "croissant", "cinnamon roll",
    "muffin", "muffins", "scone", "scones",
    # Generic dessert terms & indicators
    "dessert", "desserts", "sweet", "sweets", "mithai", "meetha", "meethi",
    "sweet dish", "after meal", "chocolate", "caramel", "butterscotch",
    "vanilla", "strawberry", "blueberry", "raspberry",
    # Candy & confectionery
    "candy", "candies", "toffee", "fudge", "praline", "nougat",
    "lollipop", "gummy", "jelly", "jellybean",
    # Specific branded/popular items
    "death by chocolate", "choco lava", "lava cake", "molten",
    "tres leches", "banoffee", "sticky toffee",
}

# Side dish keywords (expanded)
SIDE_KEYWORDS = {
    # Fries & chips (expanded)
    "fries", "french fries", "peri peri fries", "masala fries", "loaded fries",
    "potato wedges", "wedges", "chips", "nachos", "tortilla chips",
    "curly fries", "waffle fries", "cheese fries", "bacon fries",
    "sweet potato fries", "seasoned fries", "cajun fries",
    "aloo fries", "finger chips", "crinkle cut",
    # Salads (expanded extensively)
    "salad", "salads", "green salad", "garden salad", "mixed salad",
    "caesar salad", "greek salad", "russian salad", "italian salad",
    "coleslaw", "cole slaw", "slaw", "kachumber", "onion salad",
    "cucumber salad", "tomato salad", "corn salad", "pasta salad",
    "quinoa salad", "chickpea salad", "bean salad",
    # Raita & accompaniments
    "raita", "raitas", "boondi raita", "cucumber raita", "onion raita",
    "mixed raita", "aloo raita", "veg raita", "lauki raita",
    # Soups (expanded)
    "soup", "soups", "shorba", "yakhni", "rasam", "sambhar", "sambar",
    "tomato soup", "sweet corn soup", "hot and sour", "hot & sour",
    "manchow soup", "cream of mushroom", "cream of tomato", "cream soup",
    "clear soup", "veg soup", "chicken soup", "minestrone", "gazpacho",
    "thai soup", "tom kha", "miso soup", "wonton soup", "noodle soup",
    "lemon coriander", "talumein", "lung fung",
    # Bread & starters
    "garlic bread", "garlic breads", "breadstick", "breadsticks",
    "bruschetta", "crostini", "focaccia", "pita", "lavash",
    "cheese toast", "cheese bread", "butter toast",
    # Indian accompaniments
    "papad", "papadum", "papadam", "pappad", "appalam",
    "pickle", "pickles", "achaar", "achar",
    "chutney", "chutneys", "mint chutney", "tamarind chutney", "coconut chutney",
    "green chutney", "red chutney", "pudina chutney", "imli chutney",
    # Dips & sauces
    "dip", "dips", "hummus", "guacamole", "salsa", "pico de gallo",
    "mayo dip", "cheese dip", "garlic dip", "sour cream",
    "tzatziki", "baba ganoush", "labneh", "aioli",
    "sauce", "sauces", "extra sauce", "dipping sauce",
    # Appetizers & starters
    "starter", "starters", "appetizer", "appetizers", "finger food",
    "snack", "snacks", "nibbles",
    "pakora", "pakoras", "pakoda", "pakodas", "bhajiya", "bhaji", "bhajji",
    "onion rings", "onion ring", "rings",
    "mushroom", "stuffed mushroom", "mushrooms", "button mushroom",
    "paneer tikka", "hara bhara kebab", "dahi kebab", "malai chaap",
    "corn", "sweet corn", "baby corn", "corn on cob", "american corn",
    "honey chilli potato", "chilli potato", "crispy potato",
    "spring roll", "spring rolls", "veg spring roll",
    "veg roll", "crispy roll", "crunchy roll",
    "cutlet", "cutlets", "aloo cutlet", "veg cutlet",
    "tikki", "aloo tikki", "tikka", "kabab", "kebab",
    # Rice sides (plain rice accompaniments)
    "jeera rice", "steamed rice", "plain rice", "white rice",
    "fried rice side", "pulao small", "boiled rice",
    # Bread sides (Indian breads)
    "naan", "naans", "roti", "rotis", "chapati", "chapatis", "phulka",
    "paratha", "parathas", "kulcha", "kulchas", "tandoori roti",
    "butter naan", "garlic naan", "cheese naan", "keema naan",
    "laccha paratha", "missi roti", "roomali roti", "rumali",
    "bhatura", "puri", "poori", "luchi", "bread basket",
    "basket naan", "basket roti", "naan basket", "assorted bread",
    # Extra/add-on items
    "extra", "add on", "addon", "additional", "side portion",
}

# Main course keywords (comprehensive list - expanded)
MAIN_KEYWORDS = {
    # Biryani & rice dishes (expanded)
    "biryani", "biriyani", "briyani", "biryana", "hyderabadi biryani",
    "lucknowi biryani", "dum biryani", "pot biryani", "handi biryani",
    "kacchi biryani", "pakki biryani", "awadhi biryani",
    "pulao", "pulav", "pilaf", "pilau", "tehri", "tahari",
    "khichdi", "khichri", "khichuri", "pongal rice",
    "fried rice", "veg fried rice", "chicken fried rice", "egg fried rice",
    "schezwan rice", "singapore rice", "burnt garlic rice",
    "mexican rice", "spanish rice", "chinese rice",
    # Indian curries & gravies (extensive)
    "curry", "curries", "gravy", "masala", "sabzi", "sabji", "subzi",
    "bhaji", "tarkari", "korma", "kurma", "qorma",
    "dal", "daal", "dhal", "dal makhani", "dal tadka", "dal fry", "tarka dal",
    "yellow dal", "chana dal", "moong dal", "masoor dal", "toor dal",
    "paneer", "paneer butter masala", "shahi paneer", "kadai paneer",
    "palak paneer", "paneer tikka masala", "matar paneer", "paneer bhurji",
    "paneer lababdar", "paneer pasanda", "paneer do pyaza", "handi paneer",
    "chole", "chhole", "chana masala", "punjabi chole", "amritsari chole",
    "rajma", "rajma masala", "kashmiri rajma", "punjabi rajma",
    "kadhi", "kadi", "kadhi pakora", "punjabi kadhi", "gujarati kadhi",
    "aloo", "dum aloo", "aloo gobi", "aloo matar", "jeera aloo",
    "aloo palak", "aloo baingan", "kashmiri dum aloo",
    "malai kofta", "navratan korma", "mix veg", "veg kolhapuri",
    "veg jalfrezi", "veg handi", "veg kadai", "mixed vegetable",
    "bhindi", "bhindi masala", "bhindi fry", "okra",
    "baingan", "baingan bharta", "begun", "brinjal",
    "gobi", "gobi masala", "aloo gobi", "gobhi",
    "mushroom masala", "kadai mushroom", "mushroom do pyaza",
    # Non-veg Indian mains
    "chicken", "murgh", "murg", "kukad",
    "butter chicken", "makhani", "chicken tikka", "tandoori chicken",
    "chicken curry", "chicken masala", "chicken biryani", "chicken korma",
    "chicken 65", "chicken lollipop", "chicken fry", "fried chicken",
    "chicken do pyaza", "chicken handi", "chicken kadai",
    "chicken chettinad", "chicken xacuti", "chicken cafreal",
    "mutton", "lamb", "gosht", "ghost",
    "mutton curry", "mutton biryani", "mutton rogan josh", "rogan josh",
    "mutton korma", "mutton do pyaza", "keema", "kheema", "qeema",
    "seekh kebab", "galouti kebab", "shammi kebab", "kakori kebab",
    "boti kebab", "burrah kebab", "chapli kebab",
    "nihari", "haleem", "paya", "nalli", "bhuna gosht",
    "fish", "machli", "machhi", "fish curry", "fish fry", "fish tikka",
    "tandoori fish", "fish masala", "amritsari fish",
    "prawn", "prawns", "shrimp", "jhinga", "prawn curry", "prawn masala",
    "prawn biryani", "prawn fry", "koliwada prawns",
    "egg", "anda", "egg curry", "egg biryani", "egg bhurji",
    "omelette", "omelet", "egg masala", "anda curry",
    # Chinese/Indo-Chinese (expanded)
    "noodles", "noodle", "chow mein", "chowmein", "lo mein",
    "hakka noodles", "schezwan noodles", "singapore noodles",
    "pan fried noodles", "crispy noodles", "soft noodles",
    "manchurian", "gobi manchurian", "veg manchurian", "chicken manchurian",
    "dry manchurian", "gravy manchurian",
    "chilli", "chilly", "chilli paneer", "chilli chicken", "chilli potato",
    "chilli garlic", "dragon chicken", "dragon paneer",
    "schezwan", "szechuan", "szechwan", "hot garlic", "hunan",
    "kung pao", "mongolian", "cantonese", "hong kong",
    "chopsuey", "chop suey", "american chopsuey",
    "triple rice", "triple schezwan",
    # Italian (expanded)
    "pizza", "pizzas", "margherita", "pepperoni", "farmhouse",
    "bbq chicken pizza", "paneer tikka pizza", "veggie pizza",
    "thin crust", "thick crust", "stuffed crust", "deep dish",
    "neapolitan", "sicilian",
    "pasta", "pastas", "penne", "spaghetti", "fettuccine", "linguine",
    "rigatoni", "fusilli", "farfalle", "macaroni", "mac and cheese",
    "alfredo", "arrabiata", "arrabbiata", "carbonara", "bolognese",
    "marinara", "pesto", "aglio olio", "rosa", "primavera",
    "lasagna", "lasagne", "ravioli", "gnocchi", "risotto", "cannelloni",
    # American/Fast food (expanded)
    "burger", "burgers", "hamburger", "cheeseburger", "veggie burger",
    "chicken burger", "lamb burger", "fish burger",
    "zinger", "whopper", "mcchicken", "big mac", "quarter pounder",
    "crispy burger", "double burger", "tower burger",
    "sandwich", "sandwiches", "sub", "subway", "grinder", "hoagie",
    "club sandwich", "grilled sandwich", "panini", "toast sandwich",
    "wrap", "wraps", "burrito", "burritos", "taco", "tacos",
    "quesadilla", "quesadillas", "enchilada", "enchiladas", "fajita",
    "hot dog", "hotdog", "hot dogs", "sausage",
    # Asian cuisines
    "sushi", "sashimi", "ramen", "pho", "pad thai", "tom yum",
    "dim sum", "dumpling", "dumplings", "momos", "momo", "gyoza", "wonton",
    "stir fry", "teriyaki", "tempura", "katsu", "donburi",
    "thai curry", "green curry", "red curry", "massaman",
    "laksa", "satay", "rendang",
    # South Indian mains (expanded)
    "dosa", "dosai", "masala dosa", "rava dosa", "mysore dosa", "set dosa",
    "paper dosa", "ghee dosa", "onion dosa", "podi dosa",
    "uttapam", "uttappa", "oothappam", "uthappam",
    "idli", "idly", "rava idli", "kanchipuram idli",
    "vada", "vade", "medu vada", "sambar vada", "dahi vada",
    "appam", "appams", "puttu", "pesarattu", "attu",
    "upma", "uppuma", "rava upma", "pongal", "ven pongal",
    # Street food & snacks (that are mains)
    "thali", "thalis", "meals", "full meals", "mini meals",
    "combo", "combos", "platter", "platters",
    "pav bhaji", "missal pav", "misal pav", "vada pav", "vadapav", "dabeli",
    "kathi roll", "frankie", "frankies", "shawarma", "shawarmas",
    "kebab roll", "egg roll", "chicken roll",
    "samosa", "samosas", "kachori", "kachoris", "aloo samosa",
    "pani puri", "golgappa", "puchka", "bhel", "bhel puri", "sev puri",
    "chaat", "chaats", "papdi chaat", "dahi puri", "raj kachori",
    "chole bhature", "poori bhaji", "puri bhaji",
    # North Indian mains
    "paratha", "aloo paratha", "gobi paratha", "paneer paratha",
    "stuffed paratha", "laccha paratha",
    # Generic main indicators
    "full", "regular", "large", "jumbo", "family pack", "sharing",
    "meal", "meals", "combo meal", "value meal", "box meal",
    "main course", "entree", "dinner", "lunch",
    # Size indicators that typically mean mains
    "portion", "plate", "serving",
}

# Non-vegetarian indicators
NON_VEG_KEYWORDS = {
    # Meat
    "chicken", "mutton", "lamb", "beef", "pork", "bacon", "ham",
    "meat", "keema", "kheema", "gosht", "qeema",
    # Seafood
    "fish", "prawn", "prawns", "shrimp", "crab", "lobster", "squid",
    "calamari", "oyster", "clam", "mussel", "scallop",
    "jhinga", "machli", "pomfret", "surmai", "rawas", "rohu", "hilsa",
    "salmon", "tuna", "sardine", "anchovy",
    # Egg
    "egg", "omelette", "omelet", "anda", "bhurji",
    "scrambled egg", "boiled egg", "poached egg", "sunny side",
    # Specific dishes
    "tikka", "tandoori", "kebab", "kabab", "seekh", "galouti", "shammi",
    "butter chicken", "chicken 65", "chicken lollipop",
    "rogan josh", "nihari", "haleem", "biryani chicken", "biryani mutton",
    "pepperoni", "salami", "sausage",
}

# Vegetarian indicators
VEG_KEYWORDS = {
    # Explicit veg
    "veg", "vegetarian", "veggie", "pure veg", "jain",
    "paneer", "tofu", "soya", "soy chunk", "nutri",
    # Vegetables
    "aloo", "potato", "gobi", "cauliflower", "palak", "spinach",
    "mushroom", "capsicum", "corn", "peas", "matar", "beans",
    "carrot", "tomato", "onion", "brinjal", "baingan",
    "bhindi", "okra", "lauki", "tinda", "karela",
    # Dairy (veg in Indian context)
    "cheese", "cottage cheese", "malai", "cream", "butter",
    "curd", "dahi", "yogurt", "lassi", "chaas",
    # Pulses & legumes
    "dal", "daal", "chana", "chole", "rajma", "lobia",
    "moong", "urad", "masoor", "toor", "arhar",
    # Fruits
    "fruit", "apple", "mango", "banana", "orange", "pineapple",
    "strawberry", "grape", "watermelon", "papaya",
}

# Ambiguous items that need context
AMBIGUOUS_KEYWORDS = {
    "biryani", "fried rice", "noodles", "manchurian", "momos",
    "roll", "wrap", "burger", "sandwich", "pizza", "pasta",
    "curry", "masala", "gravy", "kebab", "tikka",
    "thali", "combo", "platter", "meal",
}


# -----------------------------------------------------------------------------
# Cuisine-to-Category Fallback Mapping
# -----------------------------------------------------------------------------

CUISINE_CATEGORY_FALLBACK = {
    # Bakery/Dessert cuisines → default dessert
    "bakery": ItemCategory.DESSERT,
    "desserts": ItemCategory.DESSERT,
    "ice cream": ItemCategory.DESSERT,
    "mithai": ItemCategory.DESSERT,
    "sweets": ItemCategory.DESSERT,
    "cakeshop": ItemCategory.DESSERT,
    "cake shop": ItemCategory.DESSERT,
    "confectionery": ItemCategory.DESSERT,
    
    # Beverage cuisines → default beverage
    "beverages": ItemCategory.BEVERAGE,
    "beverage": ItemCategory.BEVERAGE,
    "juices": ItemCategory.BEVERAGE,
    "juice": ItemCategory.BEVERAGE,
    "tea": ItemCategory.BEVERAGE,
    "coffee": ItemCategory.BEVERAGE,
    "cafe": ItemCategory.BEVERAGE,
    "bubble tea": ItemCategory.BEVERAGE,
    "smoothies": ItemCategory.BEVERAGE,
    "shakes": ItemCategory.BEVERAGE,
    
    # Main course cuisines → default main
    "north indian": ItemCategory.MAIN,
    "south indian": ItemCategory.MAIN,
    "chinese": ItemCategory.MAIN,
    "indo-chinese": ItemCategory.MAIN,
    "italian": ItemCategory.MAIN,
    "pizza": ItemCategory.MAIN,
    "burger": ItemCategory.MAIN,
    "fast food": ItemCategory.MAIN,
    "biryani": ItemCategory.MAIN,
    "mughlai": ItemCategory.MAIN,
    "continental": ItemCategory.MAIN,
    "thai": ItemCategory.MAIN,
    "mexican": ItemCategory.MAIN,
    "japanese": ItemCategory.MAIN,
    "korean": ItemCategory.MAIN,
    "arabian": ItemCategory.MAIN,
    "lebanese": ItemCategory.MAIN,
    "mediterranean": ItemCategory.MAIN,
    "american": ItemCategory.MAIN,
    "seafood": ItemCategory.MAIN,
    "asian": ItemCategory.MAIN,
    "street food": ItemCategory.MAIN,
    "rolls": ItemCategory.MAIN,
    "kebabs": ItemCategory.MAIN,
    "momos": ItemCategory.MAIN,
    "wraps": ItemCategory.MAIN,
    "sandwiches": ItemCategory.MAIN,
    
    # Side-oriented cuisines → default side
    "salads": ItemCategory.SIDE,
    "soups": ItemCategory.SIDE,
    "starters": ItemCategory.SIDE,
}


# -----------------------------------------------------------------------------
# Rule-Based Categorizer
# -----------------------------------------------------------------------------

class RuleBasedCategorizer:
    """
    Rule-based item categorizer using keyword matching.
    
    Priority order: beverage > dessert > side > main
    
    Falls back to:
    1. Cuisine-based category if keywords don't match
    2. "main" as final fallback (instead of "unknown")
    """
    
    def __init__(self):
        """Initialize the categorizer with keyword sets."""
        self.beverage_keywords = BEVERAGE_KEYWORDS
        self.dessert_keywords = DESSERT_KEYWORDS
        self.side_keywords = SIDE_KEYWORDS
        self.main_keywords = MAIN_KEYWORDS
        self.non_veg_keywords = NON_VEG_KEYWORDS
        self.veg_keywords = VEG_KEYWORDS
        self.cuisine_fallback = CUISINE_CATEGORY_FALLBACK
        
        # Compile regex patterns for efficient matching
        self._compile_patterns()
    
    def _compile_patterns(self) -> None:
        """Compile regex patterns for each category."""
        def make_pattern(keywords: set) -> re.Pattern:
            # Sort by length (longer first) for greedy matching
            sorted_keywords = sorted(keywords, key=len, reverse=True)
            # Escape special characters and join with OR
            escaped = [re.escape(kw) for kw in sorted_keywords]
            pattern = r'\b(' + '|'.join(escaped) + r')\b'
            return re.compile(pattern, re.IGNORECASE)
        
        self.beverage_pattern = make_pattern(self.beverage_keywords)
        self.dessert_pattern = make_pattern(self.dessert_keywords)
        self.side_pattern = make_pattern(self.side_keywords)
        self.main_pattern = make_pattern(self.main_keywords)
        self.non_veg_pattern = make_pattern(self.non_veg_keywords)
        self.veg_pattern = make_pattern(self.veg_keywords)
    
    def _find_matches(self, text: str, pattern: re.Pattern) -> List[str]:
        """Find all keyword matches in text."""
        matches = pattern.findall(text.lower())
        return list(set(matches))
    
    def _get_cuisine_fallback(self, cuisine: Optional[str]) -> Optional[ItemCategory]:
        """
        Get category fallback based on cuisine type.
        
        Args:
            cuisine: Restaurant cuisine string
            
        Returns:
            Category based on cuisine, or None if no match
        """
        if not cuisine or not isinstance(cuisine, str):
            return None
        
        cuisine_lower = cuisine.lower().strip()
        
        # Direct match
        if cuisine_lower in self.cuisine_fallback:
            return self.cuisine_fallback[cuisine_lower]
        
        # Partial match (check if any fallback key is in cuisine)
        for key, category in self.cuisine_fallback.items():
            if key in cuisine_lower:
                return category
        
        return None
    
    def categorize(
        self, 
        item_name: str, 
        cuisine: Optional[str] = None
    ) -> CategoryResult:
        """
        Categorize an item based on its name and optionally cuisine.
        
        Uses a multi-stage approach:
        1. Keyword matching (primary)
        2. Cuisine-based fallback (secondary)
        3. Default to "main" (final fallback)
        
        Args:
            item_name: Name of the menu item
            cuisine: Optional restaurant cuisine for fallback
            
        Returns:
            CategoryResult with category, confidence, and matched keywords
        """
        if not item_name or not isinstance(item_name, str):
            # Even empty items default to main (not unknown)
            return CategoryResult(
                category=ItemCategory.MAIN,
                confidence=0.3,
                matched_keywords=["fallback"]
            )
        
        item_lower = item_name.lower().strip()
        
        # Check categories in priority order
        # Priority: beverage > dessert > side > main
        
        # 1. Check beverages first (most specific)
        beverage_matches = self._find_matches(item_lower, self.beverage_pattern)
        if beverage_matches:
            confidence = min(1.0, 0.7 + 0.1 * len(beverage_matches))
            return CategoryResult(
                category=ItemCategory.BEVERAGE,
                confidence=confidence,
                matched_keywords=beverage_matches
            )
        
        # 2. Check desserts
        dessert_matches = self._find_matches(item_lower, self.dessert_pattern)
        if dessert_matches:
            confidence = min(1.0, 0.7 + 0.1 * len(dessert_matches))
            return CategoryResult(
                category=ItemCategory.DESSERT,
                confidence=confidence,
                matched_keywords=dessert_matches
            )
        
        # 3. Check sides
        side_matches = self._find_matches(item_lower, self.side_pattern)
        main_matches = self._find_matches(item_lower, self.main_pattern)
        
        # If more side matches or equal with side-specific indicators
        if side_matches:
            # Check if it's actually a main dish with side elements
            if main_matches and len(main_matches) > len(side_matches):
                confidence = min(1.0, 0.6 + 0.1 * len(main_matches))
                return CategoryResult(
                    category=ItemCategory.MAIN,
                    confidence=confidence,
                    matched_keywords=main_matches
                )
            
            confidence = min(1.0, 0.6 + 0.1 * len(side_matches))
            return CategoryResult(
                category=ItemCategory.SIDE,
                confidence=confidence,
                matched_keywords=side_matches
            )
        
        # 4. Check mains
        if main_matches:
            confidence = min(1.0, 0.6 + 0.1 * len(main_matches))
            return CategoryResult(
                category=ItemCategory.MAIN,
                confidence=confidence,
                matched_keywords=main_matches
            )
        
        # 5. FALLBACK: Cuisine-based category
        cuisine_category = self._get_cuisine_fallback(cuisine)
        if cuisine_category:
            return CategoryResult(
                category=cuisine_category,
                confidence=0.5,
                matched_keywords=["cuisine_fallback"]
            )
        
        # 6. FINAL FALLBACK: Default to "main" (not "unknown")
        # Most unclassified items in food apps are main courses
        return CategoryResult(
            category=ItemCategory.MAIN,
            confidence=0.4,
            matched_keywords=["default_fallback"]
        )
    
    def classify_veg_status(self, item_name: str) -> VegResult:
        """
        Classify vegetarian status based on item name.
        
        Args:
            item_name: Name of the menu item
            
        Returns:
            VegResult with veg status, confidence, and matched keywords
        """
        if not item_name or not isinstance(item_name, str):
            return VegResult(
                is_veg=VegStatus.UNKNOWN,
                confidence=0.0,
                matched_keywords=[]
            )
        
        item_lower = item_name.lower().strip()
        
        # Find matches
        non_veg_matches = self._find_matches(item_lower, self.non_veg_pattern)
        veg_matches = self._find_matches(item_lower, self.veg_pattern)
        
        # Non-veg indicators take precedence
        if non_veg_matches:
            confidence = min(1.0, 0.8 + 0.05 * len(non_veg_matches))
            return VegResult(
                is_veg=VegStatus.NON_VEG,
                confidence=confidence,
                matched_keywords=non_veg_matches
            )
        
        # Check for veg indicators
        if veg_matches:
            confidence = min(1.0, 0.7 + 0.05 * len(veg_matches))
            return VegResult(
                is_veg=VegStatus.VEG,
                confidence=confidence,
                matched_keywords=veg_matches
            )
        
        # Unknown - no clear indicators
        return VegResult(
            is_veg=VegStatus.UNKNOWN,
            confidence=0.0,
            matched_keywords=[]
        )


# -----------------------------------------------------------------------------
# LLM-Based Categorizer Stub
# -----------------------------------------------------------------------------

class LLMCategorizerStub:
    """
    Stub for LLM-based item categorization.
    
    This class provides the structure for LLM-based classification
    without making actual API calls. Replace the stub methods with
    actual LLM API calls when ready to use.
    
    Supported LLM providers (structure ready):
    - OpenAI GPT-4
    - Anthropic Claude
    - Local models via Ollama
    """
    
    SYSTEM_PROMPT = """You are a food item categorizer. Given a menu item name, classify it into one of these categories:
- main: Main course dishes (biryani, pizza, burger, curry, pasta, etc.)
- side: Side dishes and accompaniments (fries, salad, soup, bread, papad, etc.)
- dessert: Sweet dishes and desserts (ice cream, cake, gulab jamun, kulfi, etc.)
- beverage: Drinks (juice, coffee, tea, shake, lassi, soda, etc.)

Also determine if the item is vegetarian:
- veg: Contains no meat, fish, or eggs
- non-veg: Contains meat, fish, or eggs
- unknown: Cannot determine

Respond in JSON format:
{"category": "main|side|dessert|beverage", "veg_status": "veg|non-veg|unknown", "confidence": 0.0-1.0}
"""
    
    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initialize LLM categorizer stub.
        
        Args:
            provider: LLM provider (openai, anthropic, ollama)
            model: Model name
            api_key: API key (from env if not provided)
            base_url: Base URL for API (for custom endpoints)
        """
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        
        logger.info(f"LLM Categorizer initialized (provider={provider}, model={model})")
        logger.warning("LLM categorizer is a STUB - no actual API calls will be made")
    
    def _build_prompt(self, item_name: str) -> str:
        """Build the prompt for LLM classification."""
        return f"Classify this menu item: {item_name}"
    
    def _parse_response(self, response: str) -> Tuple[CategoryResult, VegResult]:
        """
        Parse LLM response into CategoryResult and VegResult.
        
        Args:
            response: Raw LLM response string
            
        Returns:
            Tuple of (CategoryResult, VegResult)
        """
        # Stub implementation - would parse actual JSON response
        import json
        
        try:
            data = json.loads(response)
            
            category_map = {
                "main": ItemCategory.MAIN,
                "side": ItemCategory.SIDE,
                "dessert": ItemCategory.DESSERT,
                "beverage": ItemCategory.BEVERAGE,
            }
            
            veg_map = {
                "veg": VegStatus.VEG,
                "non-veg": VegStatus.NON_VEG,
                "unknown": VegStatus.UNKNOWN,
            }
            
            category = category_map.get(data.get("category", ""), ItemCategory.UNKNOWN)
            veg_status = veg_map.get(data.get("veg_status", ""), VegStatus.UNKNOWN)
            confidence = float(data.get("confidence", 0.5))
            
            return (
                CategoryResult(category=category, confidence=confidence, matched_keywords=["llm"]),
                VegResult(is_veg=veg_status, confidence=confidence, matched_keywords=["llm"])
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            return (
                CategoryResult(category=ItemCategory.UNKNOWN, confidence=0.0, matched_keywords=[]),
                VegResult(is_veg=VegStatus.UNKNOWN, confidence=0.0, matched_keywords=[])
            )
    
    def categorize(self, item_name: str) -> Tuple[CategoryResult, VegResult]:
        """
        Categorize item using LLM (STUB - returns unknown).
        
        In production, this would:
        1. Build prompt with item name
        2. Call LLM API
        3. Parse response
        4. Return results
        
        Args:
            item_name: Name of the menu item
            
        Returns:
            Tuple of (CategoryResult, VegResult)
        """
        # STUB: Log warning and return unknown
        logger.debug(f"LLM categorization requested for: {item_name}")
        logger.debug("Returning UNKNOWN - LLM API not implemented")
        
        # Structure for actual implementation:
        # prompt = self._build_prompt(item_name)
        # 
        # if self.provider == "openai":
        #     response = self._call_openai(prompt)
        # elif self.provider == "anthropic":
        #     response = self._call_anthropic(prompt)
        # elif self.provider == "ollama":
        #     response = self._call_ollama(prompt)
        # else:
        #     raise ValueError(f"Unknown provider: {self.provider}")
        # 
        # return self._parse_response(response)
        
        return (
            CategoryResult(
                category=ItemCategory.UNKNOWN,
                confidence=0.0,
                matched_keywords=[]
            ),
            VegResult(
                is_veg=VegStatus.UNKNOWN,
                confidence=0.0,
                matched_keywords=[]
            )
        )
    
    def _call_openai(self, prompt: str) -> str:
        """
        Call OpenAI API (STUB).
        
        Implementation would use:
        ```python
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
        ```
        """
        raise NotImplementedError("OpenAI API call not implemented")
    
    def _call_anthropic(self, prompt: str) -> str:
        """
        Call Anthropic API (STUB).
        
        Implementation would use:
        ```python
        from anthropic import Anthropic
        client = Anthropic(api_key=self.api_key)
        response = client.messages.create(
            model=self.model,
            max_tokens=256,
            system=self.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
        ```
        """
        raise NotImplementedError("Anthropic API call not implemented")
    
    def _call_ollama(self, prompt: str) -> str:
        """
        Call Ollama local API (STUB).
        
        Implementation would use:
        ```python
        import requests
        response = requests.post(
            f"{self.base_url or 'http://localhost:11434'}/api/generate",
            json={
                "model": self.model,
                "prompt": f"{self.SYSTEM_PROMPT}\n\n{prompt}",
                "stream": False,
                "format": "json"
            }
        )
        return response.json()["response"]
        ```
        """
        raise NotImplementedError("Ollama API call not implemented")
    
    def batch_categorize(
        self,
        item_names: List[str],
        batch_size: int = 10
    ) -> List[Tuple[CategoryResult, VegResult]]:
        """
        Batch categorize items (STUB).
        
        In production, this would batch API calls for efficiency.
        
        Args:
            item_names: List of item names
            batch_size: Number of items per API call
            
        Returns:
            List of (CategoryResult, VegResult) tuples
        """
        logger.warning(f"Batch LLM categorization for {len(item_names)} items - STUB")
        return [self.categorize(name) for name in item_names]


# -----------------------------------------------------------------------------
# Hybrid Categorizer
# -----------------------------------------------------------------------------

class HybridCategorizer:
    """
    Hybrid categorizer combining rule-based and LLM approaches.
    
    Uses rule-based first, falls back to LLM for low-confidence results.
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.5,
        use_llm_fallback: bool = False,
        llm_config: Optional[Dict] = None,
    ):
        """
        Initialize hybrid categorizer.
        
        Args:
            confidence_threshold: Minimum confidence to accept rule-based result
            use_llm_fallback: Whether to use LLM for low-confidence items
            llm_config: Configuration for LLM categorizer
        """
        self.confidence_threshold = confidence_threshold
        self.use_llm_fallback = use_llm_fallback
        
        # Initialize categorizers
        self.rule_based = RuleBasedCategorizer()
        
        if use_llm_fallback:
            llm_config = llm_config or {}
            self.llm = LLMCategorizerStub(**llm_config)
        else:
            self.llm = None
        
        # Statistics
        self.stats = {
            "total": 0,
            "rule_based": 0,
            "cuisine_fallback": 0,
            "default_fallback": 0,
            "llm_fallback": 0,
            "unknown": 0,
        }
    
    def categorize(
        self, 
        item_name: str,
        cuisine: Optional[str] = None
    ) -> Tuple[CategoryResult, VegResult]:
        """
        Categorize item using hybrid approach.
        
        Args:
            item_name: Name of the menu item
            cuisine: Optional restaurant cuisine for fallback
            
        Returns:
            Tuple of (CategoryResult, VegResult)
        """
        self.stats["total"] += 1
        
        # Try rule-based first (with cuisine fallback built-in)
        category_result = self.rule_based.categorize(item_name, cuisine)
        veg_result = self.rule_based.classify_veg_status(item_name)
        
        # Track fallback usage
        if category_result.matched_keywords == ["cuisine_fallback"]:
            self.stats["cuisine_fallback"] += 1
        elif category_result.matched_keywords == ["default_fallback"]:
            self.stats["default_fallback"] += 1
        elif category_result.matched_keywords == ["fallback"]:
            self.stats["default_fallback"] += 1
        
        # Check if confidence is sufficient
        if category_result.confidence >= self.confidence_threshold:
            self.stats["rule_based"] += 1
            return (category_result, veg_result)
        
        # Fall back to LLM if enabled
        if self.use_llm_fallback and self.llm is not None:
            self.stats["llm_fallback"] += 1
            llm_category, llm_veg = self.llm.categorize(item_name)
            
            # Use LLM result if better
            if llm_category.confidence > category_result.confidence:
                category_result = llm_category
            if llm_veg.confidence > veg_result.confidence:
                veg_result = llm_veg
        
        if category_result.category == ItemCategory.UNKNOWN:
            self.stats["unknown"] += 1
        
        return (category_result, veg_result)
    
    def get_stats(self) -> Dict:
        """Get categorization statistics."""
        return self.stats.copy()


# -----------------------------------------------------------------------------
# Main Processing Pipeline
# -----------------------------------------------------------------------------

def load_menu_data(file_path: Path) -> pd.DataFrame:
    """Load menu items CSV file."""
    logger.info(f"Loading menu data from: {file_path}")
    
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df):,} menu items")
    
    return df


def enrich_menu_items(
    df: pd.DataFrame,
    categorizer: HybridCategorizer,
    item_name_col: str = "item_name",
    restaurant_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Enrich menu items with category and veg flag.
    
    Args:
        df: Menu items DataFrame
        categorizer: Categorizer instance
        item_name_col: Column name containing item names
        restaurant_df: Optional restaurant DataFrame for cuisine-based fallback
        
    Returns:
        Enriched DataFrame with item_category and veg_flag columns
    """
    logger.info("Enriching menu items with category and veg flag...")
    
    # Build cuisine lookup from restaurant data if available
    cuisine_lookup = {}
    if restaurant_df is not None and 'restaurant_id' in restaurant_df.columns:
        if 'cuisine' in restaurant_df.columns:
            cuisine_lookup = dict(zip(
                restaurant_df['restaurant_id'], 
                restaurant_df['cuisine']
            ))
            logger.info(f"Loaded cuisine data for {len(cuisine_lookup):,} restaurants")
    
    # Prepare output columns
    categories = []
    veg_flags = []
    category_confidences = []
    veg_confidences = []
    
    # Process each item
    total = len(df)
    for idx, row in df.iterrows():
        item_name = row.get(item_name_col, "")
        
        # Get cuisine for this restaurant (if available)
        cuisine = None
        if 'restaurant_id' in df.columns and cuisine_lookup:
            rest_id = row.get('restaurant_id', '')
            cuisine = cuisine_lookup.get(rest_id, None)
        
        # Categorize with cuisine context
        cat_result, veg_result = categorizer.categorize(item_name, cuisine)
        
        categories.append(cat_result.category.value)
        category_confidences.append(cat_result.confidence)
        
        # Convert veg status to flag (1=veg, 0=non-veg, -1=unknown)
        veg_flags.append(veg_result.is_veg.value)
        veg_confidences.append(veg_result.confidence)
        
        # Progress logging
        if (idx + 1) % 10000 == 0:
            logger.info(f"Processed {idx + 1:,}/{total:,} items ({100 * (idx + 1) / total:.1f}%)")
    
    # Add columns to DataFrame
    df = df.copy()
    df["item_category"] = categories
    df["veg_flag"] = veg_flags
    df["category_confidence"] = category_confidences
    df["veg_confidence"] = veg_confidences
    
    logger.info(f"Enrichment complete for {total:,} items")
    
    return df


def save_enriched_data(df: pd.DataFrame, output_path: Path) -> None:
    """Save enriched DataFrame to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved enriched data to: {output_path}")


def print_summary(df: pd.DataFrame, categorizer: HybridCategorizer) -> None:
    """Print categorization summary statistics."""
    logger.info("=" * 60)
    logger.info("CATEGORIZATION SUMMARY")
    logger.info("=" * 60)
    
    # Category distribution
    logger.info("Category Distribution:")
    category_counts = df["item_category"].value_counts()
    for cat, count in category_counts.items():
        pct = 100 * count / len(df)
        logger.info(f"  - {cat}: {count:,} ({pct:.1f}%)")
    
    # Unknown percentage check
    unknown_count = category_counts.get("unknown", 0)
    unknown_pct = 100 * unknown_count / len(df)
    logger.info(f"\n*** Unknown category: {unknown_count:,} ({unknown_pct:.2f}%) ***")
    if unknown_pct < 5:
        logger.info("    [SUCCESS] Unknown rate is below 5% target!")
    else:
        logger.info("    [WARNING] Unknown rate exceeds 5% target.")
    
    # Veg/Non-veg distribution
    logger.info("\nVeg/Non-Veg Distribution:")
    veg_labels = {1: "Veg", 0: "Non-Veg", -1: "Unknown"}
    veg_counts = df["veg_flag"].value_counts()
    for flag, count in veg_counts.items():
        pct = 100 * count / len(df)
        label = veg_labels.get(flag, f"Unknown({flag})")
        logger.info(f"  - {label}: {count:,} ({pct:.1f}%)")
    
    # Confidence statistics
    logger.info("\nConfidence Statistics:")
    logger.info(f"  - Category (mean): {df['category_confidence'].mean():.2f}")
    logger.info(f"  - Category (median): {df['category_confidence'].median():.2f}")
    logger.info(f"  - Veg flag (mean): {df['veg_confidence'].mean():.2f}")
    
    # Categorizer stats
    stats = categorizer.get_stats()
    logger.info("\nCategorizer Usage:")
    for key, value in stats.items():
        if value > 0:
            logger.info(f"  - {key}: {value:,}")
    
    # Low confidence items sample (fallback items)
    fallback_items = df[df["category_confidence"] < 0.5]
    if len(fallback_items) > 0:
        logger.info(f"\nFallback Items (confidence < 0.5): {len(fallback_items):,}")
        logger.info("Sample fallback items:")
        for _, row in fallback_items.head(5).iterrows():
            item_name = str(row.get('item_name', ''))[:50]
            category = row.get('item_category', 'unknown')
            confidence = row.get('category_confidence', 0)
            logger.info(f"  - '{item_name}' -> {category} (conf: {confidence:.2f})")
    
    logger.info("=" * 60)


def run_categorization_pipeline(
    input_path: Path,
    output_path: Path,
    restaurant_path: Optional[Path] = None,
    use_llm_fallback: bool = False,
    confidence_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Run the complete categorization pipeline.
    
    Args:
        input_path: Path to input menu CSV
        output_path: Path for output enriched CSV
        restaurant_path: Optional path to restaurant CSV for cuisine-based fallback
        use_llm_fallback: Whether to use LLM for uncertain items
        confidence_threshold: Minimum confidence for rule-based results
        
    Returns:
        Enriched DataFrame
    """
    logger.info("Starting item categorization pipeline...")
    
    # Load menu data
    df = load_menu_data(input_path)
    
    # Load restaurant data for cuisine lookup (if available)
    restaurant_df = None
    if restaurant_path is None:
        # Try default path
        default_rest_path = Path("data/processed/restaurants_cleaned.csv")
        if default_rest_path.exists():
            restaurant_path = default_rest_path
    
    if restaurant_path and restaurant_path.exists():
        logger.info(f"Loading restaurant data from: {restaurant_path}")
        restaurant_df = pd.read_csv(restaurant_path)
        logger.info(f"Loaded {len(restaurant_df):,} restaurants for cuisine lookup")
    else:
        logger.warning("No restaurant data available for cuisine-based fallback")
    
    # Initialize categorizer
    categorizer = HybridCategorizer(
        confidence_threshold=confidence_threshold,
        use_llm_fallback=use_llm_fallback,
    )
    
    # Enrich data with cuisine context
    df_enriched = enrich_menu_items(df, categorizer, restaurant_df=restaurant_df)
    
    # Save results
    save_enriched_data(df_enriched, output_path)
    
    # Print summary
    print_summary(df_enriched, categorizer)
    
    logger.info("Item categorization pipeline completed successfully!")
    
    return df_enriched


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Categorize menu items into main/side/dessert/beverage",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("data/processed/menu_items_cleaned.csv"),
        help="Path to input menu items CSV",
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/processed/menu_items_enriched.csv"),
        help="Path for output enriched CSV",
    )
    
    parser.add_argument(
        "--restaurants", "-r",
        type=Path,
        default=Path("data/processed/restaurants_cleaned.csv"),
        help="Path to restaurant CSV for cuisine-based fallback",
    )
    
    parser.add_argument(
        "--use-llm-fallback",
        action="store_true",
        help="Use LLM for uncertain classifications (stub - no actual API calls)",
    )
    
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum confidence to accept rule-based result",
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        run_categorization_pipeline(
            input_path=args.input,
            output_path=args.output,
            restaurant_path=args.restaurants,
            use_llm_fallback=args.use_llm_fallback,
            confidence_threshold=args.confidence_threshold,
        )
        return 0
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 99


if __name__ == "__main__":
    sys.exit(main())
