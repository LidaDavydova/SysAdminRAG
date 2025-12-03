import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = "https://documentation.ubuntu.com/server/how-to/"

headers = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0",
}


def get_links(url=BASE_URL):
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    sidebar = soup.find("li", {"class": "toctree-l1 current has-children current-page"})
    if not sidebar:
        raise ValueError("Sidebar navigation not found!")

    links = []

    def extract_from_list(tag):
        for li in tag.find_all("li", recursive=False):
            a = li.find("a", href=True)
            if a:
                full_url = urljoin(BASE_URL, a["href"])
                links.append(full_url)

            sub = li.find("ul", recursive=False)
            if sub:
                extract_from_list(sub)

    extract_from_list(sidebar.ul)

    return sorted(set(links))


if __name__ == "__main__":
    all_links = get_links()
    print(f"Collected {len(all_links)} links:")
    for link in all_links:
        print(link)
