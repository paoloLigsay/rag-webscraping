"""
This small module fetches a webpage and parses it using BeautifulSoup. This creates a countries data frame
with a list of countries containing it's name and some necessary information.
"""

from bs4 import BeautifulSoup
import requests
from typing import Any, Literal
from dataclasses import dataclass
from supabase import create_client, Client


url: str = "https://www.scrapethissite.com/pages/simple/"
page = requests.get(url, timeout=30)
soup = BeautifulSoup(page.text, "html")

class BaseModel:
    def __init__(self, **data: Any) -> None:
        for key, value in data.items():
            setattr(self, key, value)

@dataclass
class Area:
    """
    Area of a country.
    """
    value: float
    unit: Literal["km2"]

@dataclass
class Country:
    """
    Accepted kv pairs for Country data structure.
    """
    name: str
    capital: str
    population: int
    area: Area

    def as_dict(self) -> dict:
        """
        Returning Country as a dictionary.
        """
        return {
            "name": self.name,
            "capital": self.capital,
            "population": self.population,
            "area": self.area
        }

df_countries: list[Country] = []
countries = soup.find_all("div", class_ = "country")
for country in countries:
    # Extract information from the country container
    name = country.find("h3", class_ = "country-name").text.strip()
    capital = country.find("span", class_ = "country-capital").text.strip()
    population = country.find("span", class_ = "country-population").text.strip()
    area = country.find("span", class_ = "country-area").text.strip()

    # Transform it's structure
    new_area: Area = Area(value=float(area), unit="km2")
    new_country = Country(name=name, capital=capital, population=population, area=new_area)

    # Append the transformed structure to the list
    df_countries.append(new_country)

# TODO[PAO]
# Not recommended to use url and key here, ideally move to env, just for quicker and simple testing.
supabase_url: str = ""
supabase_key: str = ""
supabase: Client = create_client(supabase_url, supabase_key)

for i, country in enumerate(df_countries):
    response = (
        supabase.table("countries")
        .insert({
            "id": i,
            "name": country.name,
            "capital": country.capital,
            "population": country.population,
            "area_value": country.area.value,
            "area_unit": country.area.unit,
        })
        .execute()
    )
