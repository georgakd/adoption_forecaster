"""
Script to update Data_Sources_Format_Summary.docx by grouping data sources by format.

This script reorganizes data sources from Confluence into separate tables
grouped by Source Format (PDF, Spreadsheet, API, Microdata, Web tables, Mixed formats).
"""

from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from typing import List, Dict, Tuple
import re
import os


# Data from Confluence page "Existing process, data sources and business logic used for each country"
DATA_SOURCES = [
    {
        "country": "Argentina",
        "data_source": "El Instituto Nacional de Estadística y Censos (INDEC) - Encuesta Nacional sobre Acceso y Uso de Tecnologías de la Información y la Comunicación (ENTIC)",
        "data_source_type": "Local",
        "source_format": "PDF Translation required",
        "release_frequency": "Yearly, Oct-Dec",
        "data_used_for": "Age x gender and education (Secondary, Tertiary). Note: for age the age groups in the data source do not match our age bands. 18-29, 30-64, 65+. Note: Different sources have been looked at such as the census - GR to have confirmation on this.",
    },
    {
        "country": "Australia",
        "data_source": "Australian Communications and Media Authority (ACMA) - Communications and media in Australia: How we use the internet / Australian Bureau of Statistics",
        "data_source_type": "Local",
        "source_format": "PDF",
        "release_frequency": "Yearly, December",
        "data_used_for": "Age x gender",
    },
    {
        "country": "Australia",
        "data_source": "Education and Work",
        "data_source_type": "Local",
        "source_format": "XLSX",
        "release_frequency": "Yearly, May",
        "data_used_for": "Age and education (Secondary, Tertiary)",
    },
    {
        "country": "Australia",
        "data_source": "OECD",
        "data_source_type": "Regional",
        "source_format": "API, CSV",
        "release_frequency": "Yearly",
        "data_used_for": "Education (Secondary, Tertiary)",
    },
    {
        "country": "Austria",
        "data_source": "Eurostat",
        "data_source_type": "Regional",
        "source_format": "API",
        "release_frequency": "Yearly",
        "data_used_for": "Age x gender and education (Secondary, Tertiary)",
    },
    {
        "country": "Belgium",
        "data_source": "Eurostat",
        "data_source_type": "Regional",
        "source_format": "API",
        "release_frequency": "Yearly",
        "data_used_for": "Age x gender and education (Secondary, Tertiary)",
    },
    {
        "country": "Brazil",
        "data_source": "Centro Regional de Estudos para o Desenvolvimento da Sociedade da Informação",
        "data_source_type": "Local",
        "source_format": "PDF report, XLSX tables, Translation needed",
        "release_frequency": "Yearly, Q4 (usually Nov-Dec)",
        "data_used_for": "Age x gender and education (Secondary, Tertiary)",
    },
    {
        "country": "Bulgaria",
        "data_source": "Eurostat",
        "data_source_type": "Regional",
        "source_format": "API",
        "release_frequency": "Yearly",
        "data_used_for": "Age x gender and education (Secondary, Tertiary)",
    },
    {
        "country": "Canada",
        "data_source": "Statistics Canada - Internet, Statistics Canada - Education",
        "data_source_type": "Local",
        "source_format": "CSV, XLSX, API",
        "release_frequency": "Yearly for most ICT & labour/education tables (some less frequent) usually Q3 or Q4",
        "data_used_for": "Age x gender and education (Secondary, Tertiary)",
    },
    {
        "country": "Chile",
        "data_source": "Ministerio de Desarrollo Social y Familia, CASEN Survey",
        "data_source_type": "Local",
        "source_format": "Microdata (SPSS/Stata), PDF,XLSX tables, Translation needed",
        "release_frequency": "Every 2-3 years Q4 (Nov-Dec)",
        "data_used_for": "Age x gender and education (Secondary, Tertiary)",
    },
    {
        "country": "China",
        "data_source": "China Internet Network Information Center",
        "data_source_type": "Local",
        "source_format": "PDF, Translation needed",
        "release_frequency": "Twice yearly (historically Jan and July)",
        "data_used_for": "Age x gender and education (Secondary, Tertiary)",
    },
    {
        "country": "Colombia",
        "data_source": "Departamento Administrativo Nacional de Estadística (DANE)",
        "data_source_type": "Local",
        "source_format": "PDF, XLSX, CSV, Microdata (XML, JSON), Translation required",
        "release_frequency": "Yearly, Q4",
        "data_used_for": "Age x gender and education (Secondary, Tertiary). Note: Age bands are larger 12-24, 25-54, 55+",
    },
    {
        "country": "Croatia",
        "data_source": "Eurostat",
        "data_source_type": "Regional",
        "source_format": "API",
        "release_frequency": "Yearly",
        "data_used_for": "Age x gender and education (Secondary, Tertiary)",
    },
    {
        "country": "Czechia",
        "data_source": "Eurostat",
        "data_source_type": "Regional",
        "source_format": "API",
        "release_frequency": "Yearly",
        "data_used_for": "Age x gender and education (Secondary, Tertiary)",
    },
    {
        "country": "Denmark",
        "data_source": "Eurostat",
        "data_source_type": "Regional",
        "source_format": "API",
        "release_frequency": "Yearly",
        "data_used_for": "Age x gender and education (Secondary, Tertiary)",
    },
    {
        "country": "Egypt",
        "data_source": "Ministry of Communications and Information Technology",
        "data_source_type": "Local",
        "source_format": "PDF, XLSX, Translation required",
        "release_frequency": "Yearly, Q1 - Q2 usually",
        "data_used_for": "Age x gender. Gender x education (Secondary, Tertiary). Note: Age bands are larger 15-24, 25-74, 75+",
    },
    {
        "country": "Egypt",
        "data_source": "CAMPAS",
        "data_source_type": "Local",
        "source_format": "PDF, Translation required",
        "release_frequency": "Varies irregular",
        "data_used_for": "Education (Secondary, Tertiary)",
    },
    {
        "country": "France",
        "data_source": "Eurostat",
        "data_source_type": "Regional",
        "source_format": "API",
        "release_frequency": "Yearly",
        "data_used_for": "Age x gender and education (Secondary, Tertiary)",
    },
    {
        "country": "Germany",
        "data_source": "Eurostat",
        "data_source_type": "Regional",
        "source_format": "API",
        "release_frequency": "Yearly",
        "data_used_for": "Age x gender and education (Secondary, Tertiary)",
    },
    {
        "country": "Ghana",
        "data_source": "Pew Research Centre",
        "data_source_type": "Multi-country",
        "source_format": "PDF",
        "release_frequency": "Varies irregular",
        "data_used_for": "Age x gender and education (Secondary, Tertiary). Note: age groups are broader",
    },
    {
        "country": "Ghana",
        "data_source": "Ghana Statistical Service -Census 2021",
        "data_source_type": "Local",
        "source_format": "PDF",
        "release_frequency": "Every 10 years",
        "data_used_for": "Age x gender. Note: Age bands are a bit smaller e.g. 25-29, 30-34",
    },
    {
        "country": "Ghana",
        "data_source": "Statsbank",
        "data_source_type": "Local",
        "source_format": "API",
        "release_frequency": "Varies irregular",
        "data_used_for": "Age x gender and education (Secondary, Tertiary)",
    },
    {
        "country": "Greece",
        "data_source": "Elstat",
        "data_source_type": "Local",
        "source_format": "XLSX, Login required, also need to email for the data",
        "release_frequency": "Yearly, Q4",
        "data_used_for": "Age x gender and education",
    },
    {
        "country": "Greece",
        "data_source": "Eurostat",
        "data_source_type": "Regional",
        "source_format": "API",
        "release_frequency": "Yearly",
        "data_used_for": "Age x gender and education (Secondary, Tertiary) used for information.",
    },
    {
        "country": "Hong Kong",
        "data_source": "Hong Kong Census and Statistics Department",
        "data_source_type": "Local",
        "source_format": "PDF, XLSX, CSV",
        "release_frequency": "Yearly, June",
        "data_used_for": "Age x gender and education",
    },
    {
        "country": "Hungary",
        "data_source": "Eurostat",
        "data_source_type": "Regional",
        "source_format": "API",
        "release_frequency": "Yearly",
        "data_used_for": "Age x gender and education (Secondary, Tertiary)",
    },
    {
        "country": "India",
        "data_source": "OECD",
        "data_source_type": "Regional",
        "source_format": "API, CSV",
        "release_frequency": "Yearly",
        "data_used_for": "Education (Secondary, Tertiary)",
    },
    {
        "country": "India",
        "data_source": "Pew Research Centre",
        "data_source_type": "Multi-country",
        "source_format": "PDF",
        "release_frequency": "Varies irregular",
        "data_used_for": "Age x gender. Note: Age bands are broader 18-29, 30-49, 50+",
    },
    {
        "country": "India",
        "data_source": "Office of the Registrar General & Census Commissioner, India",
        "data_source_type": "Local",
        "source_format": "PDF, XLSX",
        "release_frequency": "Every 10 years",
        "data_used_for": "Education (Secondary, Tertiary)",
    },
    {
        "country": "Indonesia",
        "data_source": "Asosiasi Penyelenggara Jasa Internet Indonesia",
        "data_source_type": "Local",
        "source_format": "PDF, Translation needed",
        "release_frequency": "Yearly, Q4",
        "data_used_for": "Note: Age bands are larger 13-18, 19-34, 35-54,55+",
    },
    {
        "country": "Indonesia",
        "data_source": "Statistics Indonesia (SUSENA survey)",
        "data_source_type": "Local",
        "source_format": "API (JSON), XLSX",
        "release_frequency": "Yearly, Q3- Q4",
        "data_used_for": "Education (Secondary, Tertiary)",
    },
    {
        "country": "Ireland",
        "data_source": "Eurostat",
        "data_source_type": "Regional",
        "source_format": "API",
        "release_frequency": "Yearly",
        "data_used_for": "Age x gender and education (Secondary, Tertiary)",
    },
    {
        "country": "Israel",
        "data_source": "Central Bureau of Statistics - Social Survey",
        "data_source_type": "Local",
        "source_format": "Microdata (XLS), API",
        "release_frequency": "Yearly, Q4",
        "data_used_for": "Age x gender and education (Secondary, Tertiary)",
    },
    {
        "country": "Italy",
        "data_source": "Eurostat",
        "data_source_type": "Regional",
        "source_format": "API",
        "release_frequency": "Yearly",
        "data_used_for": "Age x gender and education (Secondary, Tertiary)",
    },
    {
        "country": "Japan",
        "data_source": "Ministry of Internal Affairs and Communications",
        "data_source_type": "Local",
        "source_format": "PDF, Translation required",
        "release_frequency": "Yearly, Q3-Q4",
        "data_used_for": "Age x gender",
    },
    {
        "country": "Japan",
        "data_source": "Statistics of Japan - Census 2020",
        "data_source_type": "Local",
        "source_format": "PDF, XLSX, API",
        "release_frequency": "Every 5 years, Q3-Q4",
        "data_used_for": "Education (Secondary, Tertiary)",
    },
    {
        "country": "Japan",
        "data_source": "Statistics of Japan - National Living Standards Survey Households",
        "data_source_type": "Local",
        "source_format": "PDF, XLSX, API, Translation required",
        "release_frequency": "Yearly, Q1",
        "data_used_for": "Education (Secondary, Tertiary). Note: Age bands are a bit smaller e.g. 25-29, 30-34",
    },
    {
        "country": "Kenya",
        "data_source": "Pew Research Centre",
        "data_source_type": "Multi-country",
        "source_format": "PDF",
        "release_frequency": "Varies irregular",
        "data_used_for": "Age x gender. Note: Age bands are broader 18-29, 30-49, 50+",
    },
    {
        "country": "Kenya",
        "data_source": "DHS",
        "data_source_type": "Local",
        "source_format": "PDF",
        "release_frequency": "Varies irregular",
        "data_used_for": "Internet x age Note: Age bands are a bit smaller e.g. 25-29, 30-34",
    },
    {
        "country": "Kenya",
        "data_source": "Communications Authority of Kenya",
        "data_source_type": "Local",
        "source_format": "PDF",
        "release_frequency": "Varies irregular",
        "data_used_for": "Internet x age Note: Age bands differ 15-17, 18-24, 25-34, 35-44, 45+)",
    },
    {
        "country": "Kenya",
        "data_source": "Kenya National Bureau of Statistics - Census",
        "data_source_type": "Local",
        "source_format": "PDF, XLSX",
        "release_frequency": "Varies irregular",
        "data_used_for": "Education (Secondary, Tertiary)",
    },
    {
        "country": "Malaysia",
        "data_source": "Department of Statistics of Malaysia",
        "data_source_type": "Local",
        "source_format": "PDF, XLSX",
        "release_frequency": "Yearly, Q2",
        "data_used_for": "Education (Secondary, Tertiary). Note: Age bands are a bit smaller e.g. 25-29, 30-34",
    },
    {
        "country": "Malaysia",
        "data_source": "Malaysian Communications and Multimedia Commission",
        "data_source_type": "Local",
        "source_format": "PDF",
        "release_frequency": "Yearly, Q3-Q4",
        "data_used_for": "Education (Secondary, Tertiary)",
    },
    {
        "country": "Malaysia",
        "data_source": "UNESCO National Commission",
        "data_source_type": "Local",
        "source_format": "PDF, XLSX",
        "release_frequency": "Varies irregular",
        "data_used_for": "Education (Secondary, Tertiary)",
    },
    {
        "country": "Malaysia",
        "data_source": "Ministry of Higher Education",
        "data_source_type": "Local",
        "source_format": "PDF, XLSX",
        "release_frequency": "Varies irregular",
        "data_used_for": "Education (Secondary, Tertiary)",
    },
    {
        "country": "Mexico",
        "data_source": "Instituto Nacional de Estadística y Geografía (INEGI)",
        "data_source_type": "Local",
        "source_format": "CSV, XLSX, Microdata (DBF, XLSX), Translation required",
        "release_frequency": "Yearly, Q3-Q4",
        "data_used_for": "Age x gender and education (Secondary, Tertiary)",
    },
    {
        "country": "Morocco",
        "data_source": "Agence Nationale de Réglementation des Télécommunications",
        "data_source_type": "Local",
        "source_format": "PDF, Translation required",
        "release_frequency": "Varies irregular",
        "data_used_for": "Age x gender and education (Secondary, Tertiary). Note: age bands are larger 19 - 24, 25 - 39, 40 - 59.",
    },
    {
        "country": "Netherlands",
        "data_source": "Eurostat",
        "data_source_type": "Regional",
        "source_format": "API",
        "release_frequency": "Yearly",
        "data_used_for": "Age x gender and education (Secondary, Tertiary)",
    },
    {
        "country": "New Zealand",
        "data_source": "Statistics New Zealand - Household Use of Information Technology and Communication Technology Survey, Educational Attainment- Stats New Zealand (Census 2023)",
        "data_source_type": "Local",
        "source_format": "CSV, XLSX",
        "release_frequency": "Every 6 years",
        "data_used_for": "Education",
    },
    {
        "country": "New Zealand",
        "data_source": "Education Counts",
        "data_source_type": "Local",
        "source_format": "XLSX",
        "release_frequency": "Yearly, Q3-Q4",
        "data_used_for": "Education",
    },
    {
        "country": "New Zealand",
        "data_source": "New Zealand Work Research Institute",
        "data_source_type": "Local",
        "source_format": "PDF",
        "release_frequency": "Varies irregular",
        "data_used_for": "Gender x internet",
    },
    {
        "country": "Nigeria",
        "data_source": "Pew Research Centre",
        "data_source_type": "Multi-country",
        "source_format": "PDF",
        "release_frequency": "Varies irregular",
        "data_used_for": "Age x gender. Note: Age bands are broader 18-29, 30-49, 50+",
    },
    {
        "country": "Nigeria",
        "data_source": "DHS",
        "data_source_type": "Local",
        "source_format": "PDF",
        "release_frequency": "Varies irregular",
        "data_used_for": "Internet x age. Education Note: Age bands are a bit smaller e.g. 25-29, 30-34",
    },
    {
        "country": "Nigeria",
        "data_source": "MICS",
        "data_source_type": "Local",
        "source_format": "PDF, XLSX",
        "release_frequency": "Every 10 years",
        "data_used_for": "Age x gender. Note: Age bands are a bit smaller e.g. 25-29, 30-34 Education (Secondary, Tertiary)",
    },
    {
        "country": "Norway",
        "data_source": "Eurostat",
        "data_source_type": "Regional",
        "source_format": "API",
        "release_frequency": "Yearly",
        "data_used_for": "Age x gender and education (Secondary, Tertiary)",
    },
    {
        "country": "Philippines",
        "data_source": "Social Weather Stations",
        "data_source_type": "Local",
        "source_format": "PDF Needs to be paid for",
        "release_frequency": "Yearly",
        "data_used_for": "Age x gender and education (Secondary, Tertiary). Note: changed to be paid for",
    },
    {
        "country": "Philippines",
        "data_source": "Google Consumer Barometer",
        "data_source_type": "Multi-market",
        "source_format": "XLSX, web table",
        "release_frequency": "Discontinued",
        "data_used_for": "Age x gender",
    },
    {
        "country": "Poland",
        "data_source": "Eurostat",
        "data_source_type": "Regional",
        "source_format": "API",
        "release_frequency": "Yearly",
        "data_used_for": "Age x gender and education (Secondary, Tertiary)",
    },
    {
        "country": "Portugal",
        "data_source": "Eurostat",
        "data_source_type": "Regional",
        "source_format": "API",
        "release_frequency": "Yearly",
        "data_used_for": "Age x gender and education (Secondary, Tertiary)",
    },
    {
        "country": "Romania",
        "data_source": "Eurostat",
        "data_source_type": "Regional",
        "source_format": "API",
        "release_frequency": "Yearly",
        "data_used_for": "Age x gender and education (Secondary, Tertiary)",
    },
    {
        "country": "Russia",
        "data_source": "Department of Education, Science and Innovation / Statistics of the Federal State Statistics Service",
        "data_source_type": "Local",
        "source_format": "PDF, XLSX, web tables hard to access- there are a few links for this source but no longer available",
        "release_frequency": "Mostly yearly, Q3-Q4",
        "data_used_for": "Age x gender. Note: Age bands are a bit smaller e.g. 25-29, 30-34. Education (Secondary, Tertiary) - on pause until 2029",
    },
    {
        "country": "Saudi Arabia",
        "data_source": "General Authority for Statistics (ICT Access and Usage by Households and Individuals Survey)",
        "data_source_type": "Local",
        "source_format": "PDF, XLSX Translation required",
        "release_frequency": "Yearly, Q3-Q4",
        "data_used_for": "Age x Gender and nationality. Note: Age bands are a bit smaller e.g. 25-29, 30-34. Education (Secondary, Tertiary)",
    },
    {
        "country": "Saudi Arabia",
        "data_source": "Census",
        "data_source_type": "Local",
        "source_format": "XLS",
        "release_frequency": "Every 10 years",
        "data_used_for": "Nationality",
    },
    {
        "country": "Saudi Arabia",
        "data_source": "ITU DataHub",
        "data_source_type": "Multi-market",
        "source_format": "CSV",
        "release_frequency": "Varies irregular",
        "data_used_for": "Internet usage",
    },
    {
        "country": "Serbia",
        "data_source": "Eurostat",
        "data_source_type": "Regional",
        "source_format": "API",
        "release_frequency": "Yearly",
        "data_used_for": "Age x gender and education (Secondary, Tertiary)",
    },
    {
        "country": "Singapore",
        "data_source": "Infocomm Usage in Households and Individuals Survey Report",
        "data_source_type": "Local",
        "source_format": "PDF, XLSX",
        "release_frequency": "Yearly, May",
        "data_used_for": "Age x gender. Note: age bands differ 18 - 39, 40 - 59, 60+",
    },
    {
        "country": "Singapore",
        "data_source": "Department of Statistics Singapore",
        "data_source_type": "Local",
        "source_format": "CSV, JSON",
        "release_frequency": "Yearly, May",
        "data_used_for": "Age x education (Secondary, Tertiary)",
    },
    {
        "country": "South Africa",
        "data_source": "Pew Research Centre",
        "data_source_type": "Multi-country",
        "source_format": "PDF",
        "release_frequency": "Varies irregular",
        "data_used_for": "Age x gender. Note: Age bands are broader 18-29, 30-49, 50+",
    },
    {
        "country": "South Africa",
        "data_source": "World Wide Worx, Internet Access in South Africa 2017",
        "data_source_type": "Local",
        "source_format": "PDF",
        "release_frequency": "Varies, Q2-Q4",
        "data_used_for": "Education",
    },
    {
        "country": "South Africa",
        "data_source": "General Household survey",
        "data_source_type": "Local",
        "source_format": "PDF, XLSX",
        "release_frequency": "Varies, Yearly, Q3-Q4",
        "data_used_for": "Internet. Age x gender and education (Secondary, Tertiary)",
    },
    {
        "country": "South Africa",
        "data_source": "DHS (Demographic and Health Survey)",
        "data_source_type": "Local",
        "source_format": "PDF",
        "release_frequency": "Every 5-7 years",
        "data_used_for": "Age x gender Note: Age bands are a bit smaller e.g. 25-29, 30-34",
    },
    {
        "country": "South Africa",
        "data_source": "RIA (Research ICT Africa)",
        "data_source_type": "Multi-market",
        "source_format": "PDF",
        "release_frequency": "Varies irregular",
        "data_used_for": "Education",
    },
    {
        "country": "South Korea",
        "data_source": "OECD",
        "data_source_type": "Regional",
        "source_format": "API, CSV",
        "release_frequency": "Yearly",
        "data_used_for": "Education (Secondary, Tertiary)",
    },
    {
        "country": "South Korea",
        "data_source": "NIA - National Information Society Agency - Survey on the Internet Usage of Korea",
        "data_source_type": "Local",
        "source_format": "PDF, Translation required",
        "release_frequency": "Yearly, Q3-Q4",
        "data_used_for": "Age x gender and education (Secondary, Tertiary) Note: Differing age brackets (10s,20s,30s, etc.)",
    },
    {
        "country": "Spain",
        "data_source": "Eurostat",
        "data_source_type": "Regional",
        "source_format": "API",
        "release_frequency": "Yearly",
        "data_used_for": "Age x gender and education (Secondary, Tertiary)",
    },
    {
        "country": "Sweden",
        "data_source": "Eurostat",
        "data_source_type": "Regional",
        "source_format": "API",
        "release_frequency": "Yearly",
        "data_used_for": "Age x gender and education (Secondary, Tertiary)",
    },
    {
        "country": "Switzerland",
        "data_source": "Eurostat",
        "data_source_type": "Regional",
        "source_format": "API",
        "release_frequency": "Yearly",
        "data_used_for": "Age x gender and education (Secondary, Tertiary)",
    },
    {
        "country": "Switzerland",
        "data_source": "Swiss Federal Statistical Office",
        "data_source_type": "Local",
        "source_format": "PX, Translation required",
        "release_frequency": "Bi-annual",
        "data_used_for": "Age x gender",
    },
    {
        "country": "Taiwan",
        "data_source": "Ministry of Digital Affairs",
        "data_source_type": "Local",
        "source_format": "PDF",
        "release_frequency": "Yearly, Q3-Q4",
        "data_used_for": "Age x gender and education (Secondary, Tertiary)",
    },
    {
        "country": "Taiwan",
        "data_source": "Ministry of Education",
        "data_source_type": "Local",
        "source_format": "PDF",
        "release_frequency": "Yearly, Q3-Q4",
        "data_used_for": "Education",
    },
    {
        "country": "Thailand",
        "data_source": "National Statistics Office of Thailand",
        "data_source_type": "Local",
        "source_format": "PDF, XLSX",
        "release_frequency": "Yearly, Q3-Q4",
        "data_used_for": "Age x gender and education (Secondary, Tertiary). Note: Age bands are a bit smaller e.g. 25-29, 30-34",
    },
    {
        "country": "Turkey",
        "data_source": "Turkish Statistical Institute",
        "data_source_type": "Local",
        "source_format": "XLSX",
        "release_frequency": "Yearly, Q2-Q3",
        "data_used_for": "Age x gender",
    },
    {
        "country": "Turkey",
        "data_source": "Eurostat",
        "data_source_type": "Local",
        "source_format": "API",
        "release_frequency": "Yearly",
        "data_used_for": "Education (Secondary, Tertiary)",
    },
    {
        "country": "UAE",
        "data_source": "Google Consumer Barometer",
        "data_source_type": "Multi-market",
        "source_format": "XLSX, web table",
        "release_frequency": "Discontinued",
        "data_used_for": "Age x gender",
    },
    {
        "country": "UAE",
        "data_source": "Statistics Centre − Abu Dhabi (SCAD)",
        "data_source_type": "Local",
        "source_format": "PDF, XLSX sign in required",
        "release_frequency": "Yearly, Q3-Q4",
        "data_used_for": "Gender and education (Secondary, Tertiary)",
    },
    {
        "country": "UAE",
        "data_source": "Dubai Statistics Center",
        "data_source_type": "Local",
        "source_format": "PDF, XLSX",
        "release_frequency": "Yearly, Q3-Q4",
        "data_used_for": "Ethnicity x gender. Gender and education (Secondary, Tertiary)",
    },
    {
        "country": "UK",
        "data_source": "Ofcom",
        "data_source_type": "Local",
        "source_format": "PDF",
        "release_frequency": "Yearly, Q3-Q3",
        "data_used_for": "Age x gender and education (Secondary, Tertiary)",
    },
    {
        "country": "UK",
        "data_source": "ONS",
        "data_source_type": "Local",
        "source_format": "XLSX",
        "release_frequency": "Discontinued",
        "data_used_for": "Age x gender",
    },
    {
        "country": "UK",
        "data_source": "Eurostat",
        "data_source_type": "Regional",
        "source_format": "API",
        "release_frequency": "Yearly",
        "data_used_for": "No longer available past data used for trends (Age x gender and education)",
    },
    {
        "country": "UK",
        "data_source": "OECD",
        "data_source_type": "Multi-market",
        "source_format": "API, CSV",
        "release_frequency": "Yearly",
        "data_used_for": "Education (Secondary, Tertiary)",
    },
    {
        "country": "USA",
        "data_source": "US Census Bureau - Gender, age ethnicity.",
        "data_source_type": "Local",
        "source_format": "CSV/XLSX, API",
        "release_frequency": "Yearly, Q3-Q4",
        "data_used_for": "Income x Ethnicity. Gender x age x ethnicity",
    },
    {
        "country": "USA",
        "data_source": "Income",
        "data_source_type": "Local",
        "source_format": "web tables (CSV/XLSX)",
        "release_frequency": "Yearly, Q3-Q4",
        "data_used_for": "Income x Ethnicity. Gender x age x ethnicity",
    },
    {
        "country": "USA",
        "data_source": "NTIA",
        "data_source_type": "Local",
        "source_format": "PDF, CSV, microdata",
        "release_frequency": "Irregular",
        "data_used_for": "State X internet pen",
    },
    {
        "country": "USA",
        "data_source": "Pew Research Centre, Who uses the internet?",
        "data_source_type": "Local",
        "source_format": "Website (tables download CSV)",
        "release_frequency": "Irregular",
        "data_used_for": "Int Pen by Income",
    },
    {
        "country": "Vietnam",
        "data_source": "General Statistics Office of Viet Nam & UNICEF - Multiple Indicator Cluster Surveys (MICS)",
        "data_source_type": "Local",
        "source_format": "PDF, microdata",
        "release_frequency": "Every 3-5 years, Q3-Q4",
        "data_used_for": "Age x gender and education (Secondary, Tertiary)",
    },
    {
        "country": "Vietnam",
        "data_source": "Census 2019",
        "data_source_type": "Local",
        "source_format": "PDF",
        "release_frequency": "Census every 10 years, Q3-Q4",
        "data_used_for": "",
    },
    {
        "country": "Vietnam",
        "data_source": "Statistical Yearbook for Vietnam",
        "data_source_type": "Local",
        "source_format": "PDF",
        "release_frequency": "Irregular",
        "data_used_for": "Age x gender and education (Secondary, Tertiary) Note: Age bands are a bit smaller e.g. 25-29, 30-34",
    },
]


def normalize_format(format_str: str) -> str:
    """Normalize format string by removing extra whitespace and converting to lowercase."""
    return re.sub(r"\s+", " ", format_str.strip().lower())


def requires_translation(format_str: str) -> bool:
    """Check if a format string indicates translation is required."""
    normalized = normalize_format(format_str)
    return "translation" in normalized


def detect_formats(format_str: str) -> Tuple[List[str], bool]:
    """
    Detect which format types are present in the format string.
    Translation requirements are excluded from format detection.

    Returns:
        Tuple of (list of format keywords found, is_mixed)
    """
    normalized = normalize_format(format_str)
    formats_found = []

    # Check for each format type (excluding translation requirements)
    if "pdf" in normalized:
        formats_found.append("PDF")
    if "json" in normalized:
        formats_found.append("JSON")
    if "xlsx" in normalized or "csv" in normalized:
        formats_found.append("SPREADSHEET")
    if "xml" in normalized:
        formats_found.append("XML")
    if "api" in normalized:
        formats_found.append("API")
    if "px" in normalized or "spss" in normalized or "stata" in normalized or "dbf" in normalized:
        formats_found.append("MICRODATA")
    if "web table" in normalized or "web tables" in normalized or "html" in normalized:
        formats_found.append("WEB_TABLES")

    # Check if mixed (has multiple distinct format types)
    is_mixed = len(formats_found) > 1

    return formats_found, is_mixed


def get_countries_requiring_translation(data_sources: List[Dict]) -> Dict[str, List[str]]:
    """
    Extract countries and their data sources that require translation.

    Returns:
        Dictionary mapping country names to lists of data source names requiring translation
    """
    translation_required = {}

    for source in data_sources:
        if requires_translation(source["source_format"]):
            country = source["country"]
            data_source = source["data_source"]

            if country not in translation_required:
                translation_required[country] = []

            if data_source not in translation_required[country]:
                translation_required[country].append(data_source)

    return translation_required


def categorize_by_format(data_sources: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Categorize data sources by their format type.

    Returns:
        Dictionary with keys: PDF, SPREADSHEET, API, MICRODATA, WEB_TABLES, MIXED
    """
    categories = {"PDF": [], "SPREADSHEET": [], "API": [], "MICRODATA": [], "WEB_TABLES": [], "MIXED": []}

    for source in data_sources:
        format_str = source["source_format"]
        formats_found, is_mixed = detect_formats(format_str)

        if is_mixed:
            categories["MIXED"].append(source)
        elif "PDF" in formats_found and len(formats_found) == 1:
            categories["PDF"].append(source)
        elif "SPREADSHEET" in formats_found and len(formats_found) == 1:
            categories["SPREADSHEET"].append(source)
        elif "API" in formats_found and len(formats_found) == 1:
            categories["API"].append(source)
        elif "MICRODATA" in formats_found and len(formats_found) == 1:
            categories["MICRODATA"].append(source)
        elif "WEB_TABLES" in formats_found and len(formats_found) == 1:
            categories["WEB_TABLES"].append(source)
        else:
            # Fallback: if it has multiple formats but doesn't match above, put in mixed
            categories["MIXED"].append(source)

    return categories


def set_cell_border(cell, **kwargs):
    """Set border properties for a table cell."""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()

    for border_name in ["top", "left", "bottom", "right"]:
        border_val = kwargs.get(border_name)
        if border_val:
            tag = f"{border_name}"
            element = OxmlElement(f"w:{tag}")
            element.set(qn("w:val"), border_val)
            element.set(qn("w:sz"), "4")
            element.set(qn("w:space"), "0")
            element.set(qn("w:color"), "000000")
            tcPr.append(element)


def add_table_to_document(doc: Document, title: str, data: List[Dict]) -> None:
    """Add a formatted table to the document with the given title and data."""
    if not data:
        return

    # Add section heading
    doc.add_heading(title, level=2)

    # Create table
    table = doc.add_table(rows=1, cols=5)
    table.style = "Light Grid Accent 1"

    # Add header row
    header_cells = table.rows[0].cells
    headers = ["Country", "Data source", "Data source type", "Release Frequency", "Data used for"]
    for i, header in enumerate(headers):
        header_cells[i].text = header
        # Make header bold
        for paragraph in header_cells[i].paragraphs:
            for run in paragraph.runs:
                run.font.bold = True

    # Add data rows
    for source in data:
        row_cells = table.add_row().cells
        row_cells[0].text = source["country"]
        row_cells[1].text = source["data_source"]
        row_cells[2].text = source["data_source_type"]
        row_cells[3].text = source["release_frequency"]
        row_cells[4].text = source["data_used_for"]

    # Add spacing after table
    doc.add_paragraph()


def add_translation_section(doc: Document, translation_required: Dict[str, List[str]]) -> None:
    """Add a section listing countries and data sources requiring translation."""
    if not translation_required:
        return

    # Add section heading
    doc.add_heading("Countries Requiring Translation", level=2)

    # Create table
    table = doc.add_table(rows=1, cols=2)
    table.style = "Light Grid Accent 1"

    # Add header row
    header_cells = table.rows[0].cells
    headers = ["Country", "Data Sources Requiring Translation"]
    for i, header in enumerate(headers):
        header_cells[i].text = header
        # Make header bold
        for paragraph in header_cells[i].paragraphs:
            for run in paragraph.runs:
                run.font.bold = True

    # Add data rows
    for country in sorted(translation_required.keys()):
        row_cells = table.add_row().cells
        row_cells[0].text = country
        row_cells[1].text = "; ".join(translation_required[country])

    # Add spacing after table
    doc.add_paragraph()


def create_document() -> Document:
    """Create the Word document with format-grouped tables."""
    doc = Document()

    # Title
    title = doc.add_heading("Data Sources Format Summary", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Introduction
    intro = doc.add_paragraph(
        "This document organizes data sources used for quota research by their source format. "
        "Each section contains a table listing all data sources that use a specific format type. "
        'Sources using multiple format types are grouped in the "Mixed Formats" section.'
    )
    intro_format = intro.paragraph_format
    intro_format.space_after = Pt(12)

    # Categorize data sources
    categories = categorize_by_format(DATA_SOURCES)

    # Add tables for each category
    category_titles = {
        "PDF": "PDF format",
        "SPREADSHEET": "Spreadsheet format (XLSX/CSV)",
        "API": "API format",
        "MICRODATA": "Microdata format",
        "WEB_TABLES": "Web Tables format",
        "MIXED": "Mixed format",
    }

    for category_key, title in category_titles.items():
        if categories[category_key]:
            add_table_to_document(doc, title, categories[category_key])

    # Add translation requirements section
    translation_required = get_countries_requiring_translation(DATA_SOURCES)
    add_translation_section(doc, translation_required)

    return doc


def main():
    """Main function to create and save the document."""
    doc = create_document()
    output_path = os.path.join(os.path.dirname(__file__), "Data_Sources_Format_Summary.docx")
    doc.save(output_path)
    print(f"Document created successfully at {output_path}")


if __name__ == "__main__":
    main()
