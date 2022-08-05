import requests
from bs4 import BeautifulSoup

def get_words(list_num=1, word_length=5):
    first_num = ((list_num - 1) * 1000) + 1
    last_num = first_num + 999
    num_str = f'{first_num}-{last_num}'
    url = f'https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists/TV/2006/{num_str}'
    page = requests.get(url)
    soup = BeautifulSoup(page.content)
    table = soup.findAll('table')[0].tbody
    words = []
    for row in table.findAll('tr')[1:]:
        word = row.findAll('td')[1].contents[0].text
        word = ''.join([x for x in word if x.isalpha()])
        if word != word.lower(): continue
        if (word_length is None) or (len(word) == word_length):
            words.append(word)
    return words

if __name__ == '__main__':
    words = get_words()
    with open('words.txt', 'w+') as fp:
        for word in words:
            fp.write(word + '\n')
