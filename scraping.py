import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from dateutil import parser
import re
import google_colab_selenium as gs
import time
from selenium.webdriver.support.ui import WebDriverWait
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
import base64
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By


def save_headlines_to_file(headlines, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for entry in headlines:
            date = entry['date']
            headline = entry['headline']
            file.write(f"{date} - {headline}\n")

def parse_date_from_auxiliary(date_string):
  # Extract date and time using regular expressions
  match = re.search(r'(\d+)\. (\w+) (\d+), (\d+\.\d+) Uhr', date_string)
  if match:
      day, month_str, year, time_str = match.groups()

      # Map month names to month numbers
      month_dict = {
          'Januar': 1, 'Februar': 2, 'März': 3, 'April': 4,
          'Mai': 5, 'Juni': 6, 'Juli': 7, 'August': 8,
          'September': 9, 'Oktober': 10, 'November': 11, 'Dezember': 12
      }

      month = month_dict.get(month_str)

      if month:
          # Create a datetime object
          formatted_date_string = f"{year}-{month:02d}-{day} {time_str}"
          parsed_date = datetime.strptime(formatted_date_string, '%Y-%m-%d %H.%M')

          return parsed_date
      else:
          print("Invalid month:", month_str)
  else:
      print("Invalid date format:", date_string)

def scrape_spiegel_headlines(start_date, end_date):
    base_url = 'https://www.spiegel.de/politik/deutschland'
    data = []

    page_num = 1

    parsed_date = datetime.now()
    while parsed_date >= start_date:
      url = f'{base_url}/p{page_num}/'
      response = requests.get(url)
      if response.status_code == 200:
          soup = BeautifulSoup(response.content, 'html.parser')
          articles = soup.find_all('article', {'aria-label': True})

          for article in articles:

              date_span = article.find('span', {'data-auxiliary': True})
              if date_span:
                      # Parse the date string using dateutil.parser
                      parsed_date = parse_date_from_auxiliary(str(date_span))
              else:
                # dates in different format for newest headline articles
                parsed_date = datetime.now()
              if parsed_date <= end_date and parsed_date >= start_date:

                headline = article['aria-label']
                data.append({'date': parsed_date, 'headline': headline})
      page_num += 1
    return data

def scrape_bild_headlines(start_date, end_date):
    base_url = 'https://www.bild.de/themen/uebersicht/archiv/archiv-82532020.bild.html'
    data = []

    current_date = start_date
    while current_date <= end_date:

      formatted_date = current_date.strftime('%Y-%m-%d')

      url = f'{base_url}?archiveDate={formatted_date}'
      response = requests.get(url)

      if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        articles = soup.find_all('article')
        for article in articles:
            article_class =  article.find('span', class_='stage-feed-item__channel').get_text(strip=True)
            if article_class == "Inland" or article_class == "News":
              soup = BeautifulSoup(response.content, 'html.parser')
              headline = article.find('span', class_='stage-feed-item__headline').get_text(strip=True)
              data.append({'date': current_date, 'headline': headline})

      current_date += timedelta(days=1)

    return data



def scrape_sz_headlines(start_date, end_date):
    # Set up the webdriver
    driver = gs.Chrome()

    # Navigate to the webpage
    driver.get("https://www.sueddeutsche.de/thema/Politik_Bayern")

    article_time = datetime.now()

    sz_headlines = []

    while article_time >= start_date:
        # Find and click the "Load More Articles" button
        load_more_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Mehr Artikel laden')]")

        driver.execute_script("arguments[0].click();", load_more_button)
        times = driver.find_elements(By.XPATH, "//time")

        article_time = times[-1].get_attribute("datetime")
        article_time = datetime.strptime(article_time, "%Y-%m-%d")

        headlines = driver.find_elements(By.XPATH, "//span[@class='css-1bdtvj4' and @data-manual='teaser-title']")
        for i in range(len(headlines)):


            date = times[i].get_attribute("datetime")
            date = datetime.strptime(date, "%Y-%m-%d")

            if date >= start_date and date <=end_date:
                sz_headlines.append({'date': date, 'headline': headlines[i].get_attribute("innerText")})
        # Wait for some time to let the content load
        time.sleep(2)

    # Close the webdriver
    driver.quit()
    return sz_headlines


def scrape_faz_headlines():
    base_url = "https://fazarchiv.faz.net/faz-portal/faz-archiv"
    query_params = {
        'q': 'WORTE:>1',
        'source': '',
        'max': 10,
        'sort': '',
        'offset': 0,  # Initial offset
        'searchIn': 'TI',
        '_ts': 1702210446712,
        'KO': 'Politik',
        'DT_from': '07.09.2023',
        'DT_to': '07.10.2023',
        'timeFilterType': 0,
        'CN':'C4EUGE'
    }

    url_parts = list(urlparse(base_url))
    url_parts[4] = urlencode(query_params)
    data = []

    offset = 0

    parsed_date = datetime.now()
    for offset in range(0, 2500, 10):
      url_parts[4] = urlencode({**query_params, 'offset': offset})
      url = urlunparse(url_parts)
      response = requests.get(url)
      if response.status_code == 200:
          soup = BeautifulSoup(response.content, 'html.parser')

          # Find all <span> tags with class "priceButton"
          price_buttons = soup.find_all('span', class_='priceButton')
          # Every article has two price buttons
          for i in range(0, len(price_buttons), 2):
              a_tag = price_buttons[i].find_previous('a')
              a_tag2 = a_tag.find_previous('a',href="#", rel="nofollow", class_=lambda c: c != 'hit-link-button')
              headline = a_tag2.text.strip()

              data.append({'date': "-", 'headline': headline})
      time.sleep(10)

    return data

def scrape_focus_headlines(start_nr, end_nr):
    base_url = 'https://www.focus.de/magazin/archiv/jahrgang_2023'
    data = []
    ausgaben_nr = start_nr

    current_date = datetime.now()
    while ausgaben_nr <= end_nr:

      formatted_date = current_date.strftime('%Y-%m-%d')

      url = f'{base_url}/ausgabe_{ausgaben_nr}/'

      response = requests.get(url)

      if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find date
        date_div = soup.find('div', class_='box-hd-title')
        if date_div:
            date_text = date_div.get_text(strip=True)

            date_parts = date_text.split('vom')
            if len(date_parts) == 2:
                actual_date = date_parts[1].strip()
        headline_div = soup.find('div', class_='bgc-b2 f-m bld pd-l10 pd-t4 pd-b3')

        if headline_div:

            for sibling in headline_div.find_next_siblings():
                # ONLY headlines from political section
                if sibling.name == 'table':
                    headline_tags = sibling.find_all('a')
                    for headline_tag in headline_tags:
                        hl = headline_tag.find_all('b')
                        if hl:
                          headline= headline_tag.get_text(strip=True, separator=' ')
                          data.append({'date': actual_date, 'headline': headline})

                elif sibling.name == 'div' and 'bgc-b2' in sibling['class']:
                    # Stop when reaching the next headline div
                    break

      ausgaben_nr +=1

    return data



def scrape_tagesschau_headlines(start_date, end_date):
    base_url = 'https://www.tagesschau.de/archiv'
    data = []

    current_date = start_date
    while current_date <= end_date:

      formatted_date = current_date.strftime('%Y-%m-%d')

      url = f'{base_url}?datum={formatted_date}&filter=inland'

      response = requests.get(url)

      if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')


        span_list = soup.find_all('div', class_='copytext-element-wrapper__vertical-only')
        for element in span_list:
              topline = element.find('span',  class_='teaser-right__labeltopline')
              headline = element.find('span', class_='teaser-right__headline')

              if topline is not None and headline is not None:
                data.append({'date': current_date, 'headline': topline.text + ': ' + headline.text})
              elif headline is not None:
                data.append({'date': current_date, 'headline': headline.text})

      current_date += timedelta(days=1)

    return data

def scrape_welt_headlines(start_date, end_date):
    base_url = 'https://www.welt.de/schlagzeilen'
    data = []

    current_date = start_date
    while current_date <= end_date:

      formatted_date = current_date.strftime('%d-%m-%Y')

      url = f'{base_url}/nachrichten-vom-{formatted_date}.html'


      response = requests.get(url)

      if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        articles = soup.find_all('article')
        for article in articles:

              section = None
              badge = article.find('a', class_=lambda text: 'badge' in text.lower())
              if badge:
                section = badge.get('title', '')
              if section == "politik":
                headline_a = article.find('a', class_=lambda text: 'headline' in text.lower())
                if headline_a:
                  headline = headline_a.get('title', '')
                  data.append({'date': current_date, 'headline': headline})

      current_date += timedelta(days=1)

    return data


def scrape_t_online_headlines(start_date, end_date):
    base_url = 'https://www.t-online.de/nachrichten/deutschland'
    article_date = datetime.now()
    page_num = 1
    data = []
    # Navigate to the news website
    while article_date >= start_date:
        print(article_date)
        print(page_num)
        driver = gs.Chrome()
        url = f'{base_url}/page_{page_num}/'
        driver.get(url)


        # Find all the article links using XPath
        article_links = driver.find_elements(By.XPATH, '//a[@class="css-1735wak" and @data-tb-link="true"]')

        # Loop through each article link
        for i in range(len(article_links)):
            # Get the href attribute of the article link
            article_url = article_links[i].get_attribute("href")

            response = requests.get(article_url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                spans = soup.find('span', class_ = "css-1qrq0jm")

                try:
                  date_format = "Aktualisiert am %d.%m.%Y"
                  article_date = datetime.strptime(spans.text, date_format)

                except:
                  try:
                    date_string, _  = spans.text.split(' - ')
                    date_format = "Aktualisiert am %d.%m.%Y"

                    article_date = datetime.strptime(date_string, date_format)
                  except:
                    try:
                      date_string, _  = spans.text.split(' - ')
                      date_format = "%d.%m.%Y"

                      article_date = datetime.strptime(date_string, date_format)
                    except:
                      date_format = "%d.%m.%Y"
                      article_date = datetime.strptime(spans.text, date_format)
                if article_date >= start_date and article_date <= end_date:
                    headline_div = soup.find('div', {'class': 'css-1uii5kk', 'data-external-article-headline': True})

                    # Extract the text content of the div element
                    headline_text = headline_div.text if headline_div else None
                    if headline_text is None:
                                  # Find the div element with the specified class and attribute
                        headline_div = soup.find('div', {'class': 'css-1a8fqf0', 'data-external-article-headline': True})

                        # Extract the text content of the div element and unescape it
                        headline_text = headline_div.text if headline_div else None

                    data.append({'date': article_date, 'headline': headline_text})

        page_num += 1
        # Close the web driver
        driver.quit()
    return data


def scrape_zeit_headlines(start_nr, end_nr):
    base_url = 'https://www.zeit.de/2023'
    data = []
    ausgaben_nr = start_nr
    while ausgaben_nr <= end_nr:
      url = f'{base_url}/{ausgaben_nr}/index'

      response = requests.get(url)

      if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        divs = soup.find_all("div", class_ = 'cp-region cp-region--solo')
        for div in divs:

          section = div.find('h2', class_='cp-area__headline')
          if section:

            if section.text == "Politik":
              # Find headlines
                headline_tags_big = div.find_all('a', class_ = "teaser-large__faux-link")
                for headline_tag in headline_tags_big:

                  if headline_tag is not None:

                      inner_response = requests.get(headline_tag.get("href"))
                      inner_soup = BeautifulSoup(inner_response.content, 'html.parser')
                      date_span = inner_soup.find('span', class_ = "encoded-date").get("data-obfuscated")
                      # Decode the obfuscated data
                      decoded_data = base64.b64decode(date_span).decode("utf-8")

                      # Parse the decoded string into a datetime object

                      if "Oktober" in decoded_data:
                        decoded_data = decoded_data.replace("Oktober", "October")

                      article_date = datetime.strptime(decoded_data, "%d. %B %Y, %H:%M Uhr")
                      article_date =  article_date.strftime("%Y-%m-%d %H:%M:%S")

                      headline= headline_tag.get_text(strip=True, separator=' ')

                      data.append({'date': article_date, 'headline': headline})

                headline_tags_small = div.find_all('a', class_ = "teaser-small__faux-link")
                for headline_tag in headline_tags_small:

                      headline= headline_tag.get_text(strip=True, separator=' ')
                      data.append({'date': article_date, 'headline': headline})

      ausgaben_nr +=1

    return data


def scrape_n_tv_headlines(start_date, end_date):
    base_url = 'https://www.n-tv.de/suche/'
    article_date = datetime.now()
    page_num = 1
    data = []
    chrome_options = Options()
    chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
    driver = gs.Chrome( options=chrome_options)

    # Navigate to the news website
    while article_date >= start_date:
        print(page_num)
        url = f'{base_url}?q=2023&at=m&page={page_num}'
        driver.get(url)
        print(url)

        # Use Selenium to interact with the page
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        articles = soup.find_all('article')

        for article in articles:
            date_span = article.find('span', class_='teaser__date')
            date_string = date_span.text.split(' ')[0].split('\n')

            try:
                date_format = "%d.%m.%Y"
                article_date = datetime.strptime(date_string[0], date_format)
            except:
                try:
                    date_format = "%d.%m.%Y"
                    article_date = datetime.strptime(date_string[1], date_format)
                except:
                    print(date_string)

            print(article_date)
            headline_span = article.find('span', class_='teaser__headline')
            headline = headline_span.text
            print(headline)

            if article_date >= start_date and article_date <= end_date:
                data.append({'date': article_date, 'headline': headline})
                print()

        page_num += 1
        time.sleep(3)
    # Close the browser window
    driver.quit()

    return data