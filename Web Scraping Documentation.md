<h1>Welcome!<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Miku-Waterfront---Vancouver-BC" data-toc-modified-id="Miku-Waterfront---Vancouver-BC-1"><a href="https://mikurestaurant.com" target="_blank"><span style="color: darkred">Miku Waterfront - Vancouver BC</span></a></a></span></li><li><span><a href="#1.-Import-libraries" data-toc-modified-id="1.-Import-libraries-2">1. Import libraries</a></span></li><li><span><a href="#2.-Establish-connection-to-the-web-page" data-toc-modified-id="2.-Establish-connection-to-the-web-page-3">2. Establish connection to the web page</a></span></li><li><span><a href="#3.-Identify-patterns-of-the-web-page" data-toc-modified-id="3.-Identify-patterns-of-the-web-page-4">3. Identify patterns of the web page</a></span><ul class="toc-item"><li><span><a href="#3.1-How-many-reviews-are-there?" data-toc-modified-id="3.1-How-many-reviews-are-there?-4.1">3.1 How many reviews are there?</a></span></li><li><span><a href="#3.2-How-many-pages-of-reviews-are-there?" data-toc-modified-id="3.2-How-many-pages-of-reviews-are-there?-4.2">3.2 How many pages of reviews are there?</a></span></li><li><span><a href="#3.3-How-many-reviews-are-there-on-each-page?" data-toc-modified-id="3.3-How-many-reviews-are-there-on-each-page?-4.3">3.3 How many reviews are there on each page?</a></span></li></ul></li><li><span><a href="#4.-Generate-variables-and-scrape-important-information" data-toc-modified-id="4.-Generate-variables-and-scrape-important-information-5">4. Generate variables and scrape important information</a></span><ul class="toc-item"><li><span><a href="#4.1-What-is-the-username?" data-toc-modified-id="4.1-What-is-the-username?-5.1">4.1 What is the username?</a></span></li><li><span><a href="#4.2-Where-is-the-user-from?" data-toc-modified-id="4.2-Where-is-the-user-from?-5.2">4.2 Where is the user from?</a></span></li><li><span><a href="#4.3-What-is-the-comment?" data-toc-modified-id="4.3-What-is-the-comment?-5.3">4.3 What is the comment?</a></span></li><li><span><a href="#4.4-When-is-the-review-posted?" data-toc-modified-id="4.4-When-is-the-review-posted?-5.4">4.4 When is the review posted?</a></span></li><li><span><a href="#4.5-What-is-the-rating-for-the-restaurant-given-by-the-user?" data-toc-modified-id="4.5-What-is-the-rating-for-the-restaurant-given-by-the-user?-5.5">4.5 What is the rating for the restaurant given by the user?</a></span></li></ul></li><li><span><a href="#5.--Extract-all-reviews-and-export-data-to-a-csv.-file" data-toc-modified-id="5.--Extract-all-reviews-and-export-data-to-a-csv.-file-6">5.  Extract all reviews and export data to a csv. file</a></span></li></ul></div>

# <span style="color:black">Web Scraping Documentation</span>
### [<span style="color:darkred">Miku Waterfront - Vancouver BC</span>](https://mikurestaurant.com)
<br/>          

**Roy Wu**      
**June, 2021**  

<br/>



**Hello there!** Welcome to my Github repository! This documentation displays and explains the code I used for data scraping.   

Although we are able to find lots of information on the Internet nowadays, and that many datasets are widely available on platforms such as Kaggle, it is sometimes useful to collect data ourselves for the specific topic that we are interested in. The purpose of this web scrapping project is to utilize libraries in Python to collect  publicly available data. Since [Miku Vancouver](https://mikurestaurant.com) is one of my favourite restaurants, and that I am passionate about the hospitality industry, we will go through how I scraped [Yelp](https://www.yelp.ca/biz/miku-vancouver-2) customer reviews for this restaurant. The goal for this project is to get started on real-world Data Science problem solving. 

### 1. Import libraries


```python
# BeautifulSoup and Requests are Python packages to extract data from web pages
from bs4 import BeautifulSoup
import requests

# Regular expression operations will be used to identify specified search pattern for strings
import re
```



### 2. Establish connection to the web page


```python
# Assigning the URL of the web page to the variable 'miku'
miku = 'https://www.yelp.ca/biz/miku-vancouver-2'

# Assigning the result of a request of the web page to the variable 'req'
req = requests.get(miku).text

# Assigning Python’s html.parser and passing the file to the BeautifulSoup constructor to establish the connection
text = BeautifulSoup(req, 'html.parser')
```



### 3. Identify patterns of the web page

#### 3.1 How many reviews are there?


```python
# Enable developer tools on the browser to identify HTML elements
# Locate and retrieve the number of reviews for Miku

# The number of review is included inside a <span> element with the 'css-bq71j2' class
review_count = text.find("span", attrs={'class': "css-bq71j2"}).string

# Since we only want to return the integer of the number of reviews,
# we use \d in Regex to match any decimal digit and return as a list 
# of strings by using 'findall()'
review_count = int(re.findall('\d+', review_count)[0])


print("There are " + str(review_count) + " reviews for Miku.")
```

#### 3.2 How many pages of reviews are there?


```python
# Create a list of web pages and and assign the variable "url_list"
url_list = []

# By going through each page, we see that the ending of the page appears
# to be numbers and they increment by 10 for each following page
# The loop identifies all the web pages and appends them in to the url list
for page in range(0, review_count, 10):
    url_list.append(
        'https://www.yelp.ca/biz/miku-vancouver-2?start=' + str(page))
print("There are " + str(len(url_list)) + " pages of reviews for Miku.")

print("\nFirst 5 Pages: ")
for url in url_list[0:5]:
    print(url)
```

#### 3.3 How many reviews are there on each page?


```python
# Identify the number of reviews on each page
page_count = text.find_all("div", attrs = {'class': 'review__373c0__13kpL border-color--default__373c0__2oFDT'})
print("There are " + str(len(page_count)) + " reviews on each page.")
```



### 4. Generate variables and scrape important information

In order to detect and eliminate potential errors, information on one review will be examined first. If successful, same process will be applied to a loop to extract all reviews.

#### 4.1 What is the username?


```python
page = page_count[0]

# Username
username = page.find('a', attrs = {'class': 'css-166la90'}).string
print(username)
```

    Justin K.


#### 4.2 Where is the user from?


```python
# Location
location = page.find('span', attrs = {'class':  'css-n6i4z7'}).get_text()
print(location)
```

    Richmond, BC




#### 4.3 What is the comment?


```python
# Comment
comment = page.find('span', attrs = {'class': "raw__373c0__3rcx7"}).get_text()
print(comment)
```

    Went for New Year's Eve dinner and as always, Miku did not disappoint. It wasn't their fault but due to COVID restrictions, Bonnie Henry just announced no alcohol after 8:00pm so luckily we reserved our table at 7:00pm. We decided to go with the Waterfront Kaiseki for $88 which included 5 dishes. We started off with their hamachi crudo, chef's selection sashimi, baked saikyo miso sablefish, miku signature sushi selection and ended with their green tea opera. Everything was delicious and experience was top notch as always.As I've been here many times, the experience never gets old and the service is always a big reason why I come back. I will definitely come back when the restrictions get better as the waterfront view never disappoints!


#### 4.4 When is the review posted?


```python
# Date
date = page.find('span', attrs = {'class': "css-e81eai"}).get_text()
print(date)
```

    3/16/2021


#### 4.5 What is the rating for the restaurant given by the user?


```python
# Rating

for rating in page.select('div[class*="i-stars__373c0__1T6rz"]'):
    rating = rating['aria-label']
    print(rating)
    
```

    5 star rating


Extracting data on the first review is successful! Now we will proceed to the next step to extract all reviews!



### 5.  Extract all reviews and export data to a csv. file


```python
import csv

# Creating a function named 'scraping' to extract all information and return a csv file
def scraping(page_count, csvwriter):
    for page in page_count:
        dataframe = {}
        username = page.find('a', attrs={'class': 'css-166la90'}).string
        location = page.find('span', attrs={'class':  'css-n6i4z7'}).get_text()
        comment = page.find(
            'span', attrs={'class': "raw__373c0__3rcx7"}).get_text()
        date = page.find('span', attrs={'class': "css-e81eai"}).get_text()
        for rating in page.select('div[class*="i-stars__373c0__1T6rz"]'):
            rating = rating['aria-label']
        dataframe['username'] = username
        dataframe['location'] = location
        dataframe['comment'] = comment
        dataframe['date'] = date
        dataframe['rating'] = rating
        page_writer.writerow(dataframe.values())


# Looping through each page and return information wanted, saving csv. file as 'Miku_Review.csv'
with open('Miku_Review.csv', 'w', encoding='utf-8') as csvfile:
    page_writer = csv.writer(csvfile)
    for index, miku in enumerate(url_list):
        response = requests.get(miku).text
        soup = BeautifulSoup(response, 'html.parser')
        page_count = soup.find_all("div", attrs={
                                   'class': 'review__373c0__13kpL border-color--default__373c0__2oFDT'})
        scraping(page_count, page_writer)
```

Completed!
