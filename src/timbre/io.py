import csv


def read_csv_titles(csv_path: str) -> list[str]:
    titles = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            titles.append(f"{row['Artist Name(s)']} - {row['Track Name']}")
    return titles
