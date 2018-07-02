import requests
from bs4 import BeautifulSoup
from sklearn.naive_bayes import GaussianNB
from sklearn.externals import joblib
import pickle

# replace this path with your path to the folder these files are stored in
PATH_TO_CURRENT_FOLDER = "C:\\Programming\\Python_Folder\\what-writer-you\\"

def get_words_from_wikipedia():
    """
    If the file 10words.txt with the 10000 most common English words is not present, retrieves this list by scraping Wikipedia
    """
    page = requests.get("https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists/PG/2006/04/1-10000")
    content = page.content
    soup = BeautifulSoup(content,"html.parser")
    words = []
    tmp = soup.find_all("td")
    for td in tmp:
        if not (td.text.isdigit()) and "." not in td.text and not td.text.strip("\n").isdigit() and td.text not in ["Rank","Word","Count (per billion)\n"]:
            words.append(td.text)

    f = open(PATH_TO_CURRENT_FOLDER+"10words.txt","x")
    for w in words:
        f.write(w)
        f.write("\n")
    f.close()

def get_words_from_file():
    """
    Retrieves a list of the 10000 most common used words in English, to be used in other methods for frequency analysis
    """
    w = []
    try:
        f = open(PATH_TO_CURRENT_FOLDER+"10words.txt","r")
        for line in f:
            if line != "\n":
                w.append(line.strip("\n").lower())
        f.close()
    except:
        get_words_from_wikipedia()
        f = open(PATH_TO_CURRENT_FOLDER+"10words.txt","r")
        for line in f:
            if line != "\n":
                w.append(line.strip("\n").lower())
        f.close()
    return w

def get_word_frequencies_file(path_to_file,wordlist=None):
    """
    Params:
    path_to_file -> string to be converted to a list of word frequencies in the format the trained model uses
    wordlist -> optional parameter, list of 10000 most commonly used English words, whose frequencies will be measured
    """
    if wordlist == None:
        wordlist = get_words_from_file()
    try:
        f = open(path_to_file,"r",encoding="utf-8")
        s = f.read()
        f.close()
    except:
        f = open(path_to_file,"r",encoding="windows-1252")
        s = f.read()
        f.close()
    for c in ",.!?:;-\\[]()_":
        s = s.replace(c," ")
    s = s.lower()
    words = s.split()


    num_of_words = len(words)
    dict_words = {}
    for w in words:
        dict_words[w] = dict_words.get(w, 0) + 1
    
    frequencies = []
    for W in wordlist:
        frequencies.append(dict_words.get(W,0)/num_of_words)

    return frequencies

def get_word_frequencies_string(text_string):
    """
    Params:
    text_string -> string to be converted to a list of word frequencies in the format the trained model uses
    """
    wordlist = get_words_from_file()
    for c in ",.!?:;-\\[]()_":
        text_string = text_string.replace(c," ")
    words = text_string.split()
    num_of_words = len(words)
    dict_words = {}
    for w in words:
        dict_words[w] = dict_words.get(w, 0) + 1
    
    frequencies = []
    for W in wordlist:
        frequencies.append(dict_words.get(W,0)/num_of_words)

    return frequencies

def preprocess():
    """
    Sets up the feature_label_pairs.pkl file, which contains a list of two element lists
    in the form [[*list of word frequencies for the 10000 most common English words*],author's name]
    this file is referenced throughout the rest of the projects so it's important to call preprocess first
    """
    # loads wordlist, length is 9965
    word_list = get_words_from_file()

    authors = [
        "Alexandre_Dumas","Anton_Chekhov","Arthur_Conan_Doyle",
        "Benjamin_Franklin","Charles_Dickens","Franz_Kafka","Friedrich_Nietzsche",
        "Fyodor_Dostoyevsky","George_Elliot","Goethe","H_G_Wells",
        "Henry_D_Thoreau","Herman_Melville","Jack_London",
        "James_Joyce","Jane_Austen","Joseph_Conrad","Jules_Verne",
        "Leo_Tolstoy","Lewis_Carroll","Mark_Twain","Mary_Shelley",
        "Oscar_Wilde","Robert_L_Stevenson","Rudyard_Kipling","Victor_Hugo"
    ]

    """loads the word frequencies from files which contain the collected works
    of the 26 authors, downloaded from Project Gutenberg and stripped from irrelevant content,
    like tables of contents and the Project Gutenberg pre- and postscripts"""
    feature_label_pairs = []
    for a in authors:
        feature_label_pairs.append([get_word_frequencies_file(PATH_TO_CURRENT_FOLDER+"author_data\\"+a+"_total.txt",word_list),a.replace("_"," ")])

    # saves to a .pkl file
    joblib.dump(feature_label_pairs,PATH_TO_CURRENT_FOLDER+"feature_label_pairs.pkl")

def get_model(author_indexes=list(range(26))):
    """
    Params:
    author_indexes -> decides which authors to take into consideration, 
    by default it loads the frequencies of all 26 authors considered in this project
    this is the order of authors:
    Alexandre Dumas, Anton Chekhov, Arthur Conan Doyle, Benjamin Franklin, Charles Dickens,
    Franz Kafka, Friedrich Nietzsche, Fyodor Dostoyevsky, George Elliot, Goethe, H G Wells,
    Henry D Thoreau, Herman Melville, Jack London, James Joyce, Jane Austen, Joseph Conrad,
    Jules Verne, Leo Tolstoy, Lewis Carroll, Mark Twain, Mary Shelley, Oscar Wilde, Robert L Stevenson,
    Rudyard Kipling, Victor Hugo

    Returns a model for predicting which of some classic authors a given word frequency list is closest to
    """
    try:
        feature_label_pairs = joblib.load(PATH_TO_CURRENT_FOLDER+"feature_label_pairs.pkl")
    except:
        preprocess()
        feature_label_pairs = joblib.load(PATH_TO_CURRENT_FOLDER+"feature_label_pairs.pkl")
    
    features, labels = [], []
    for i in author_indexes:
        features.append(feature_label_pairs[i][0])
        labels.append(feature_label_pairs[i][1])
    clf = GaussianNB()
    clf.fit(features,labels)
    return clf

def predict_string(text_input, clf=None):
    """
    Params:
    text_input -> string to be analysed
    clf -> GaussianNB classifier model to be used in the prediction

    Predicts the classic author a string is closest to based on word frequency
    """
    frequencies = get_word_frequencies_string(text_input)
    if clf == None:
        clf = get_model()
    freqs,maxindex = clf.predict_log_proba([frequencies]),0
    for fi in range(26):
        if freqs[0][fi] > freqs[0][maxindex]:
            maxindex = fi
    authors = [
        "Alexandre_Dumas","Anton_Chekhov","Arthur_Conan_Doyle",
        "Benjamin_Franklin","Charles_Dickens","Franz_Kafka","Friedrich_Nietzsche",
        "Fyodor_Dostoyevsky","George_Elliot","Goethe","H_G_Wells",
        "Henry_D_Thoreau","Herman_Melville","Jack_London",
        "James_Joyce","Jane_Austen","Joseph_Conrad","Jules_Verne",
        "Leo_Tolstoy","Lewis_Carroll","Mark_Twain","Mary_Shelley",
        "Oscar_Wilde","Robert_L_Stevenson","Rudyard_Kipling","Victor_Hugo"
    ]
    return authors[maxindex]

def predict_string_top2(text_input, clf=None):
    """
    Params:
    text_input -> string to be analysed
    clf -> GaussianNB classifier model to be used in the prediction

    Predicts the two classic authors a string is closest to based on word frequency
    """
    frequencies = get_word_frequencies_string(text_input)
    if clf == None:
        clf = get_model()
    freqs,maxindex,secondmax = clf.predict_log_proba([frequencies]),0,0
    for fi in range(26):
        if freqs[0][fi] > freqs[0][maxindex]:
            secondmax = maxindex
            maxindex = fi
        elif freqs[0][fi] > freqs[0][secondmax]:
            secondmax = fi
    authors = [
        "Alexandre_Dumas","Anton_Chekhov","Arthur_Conan_Doyle",
        "Benjamin_Franklin","Charles_Dickens","Franz_Kafka","Friedrich_Nietzsche",
        "Fyodor_Dostoyevsky","George_Elliot","Goethe","H_G_Wells",
        "Henry_D_Thoreau","Herman_Melville","Jack_London",
        "James_Joyce","Jane_Austen","Joseph_Conrad","Jules_Verne",
        "Leo_Tolstoy","Lewis_Carroll","Mark_Twain","Mary_Shelley",
        "Oscar_Wilde","Robert_L_Stevenson","Rudyard_Kipling","Victor_Hugo"
    ]
    return [authors[maxindex],authors[secondmax]]

def predict_string_top3(text_input, clf=None):
    """
    Params:
    text_input -> string to be analysed
    clf -> GaussianNB classifier model to be used in the prediction

    Predicts the three classic authors a string is closest to based on word frequency
    """
    frequencies = get_word_frequencies_string(text_input)
    if clf == None:
        clf = get_model()
    freqs,maxindex,secondmax,thirdmax = clf.predict_log_proba([frequencies]),0,0,0
    for fi in range(26):
        if freqs[0][fi] > freqs[0][maxindex]:
            thirdmax = secondmax
            secondmax = maxindex
            maxindex = fi
        elif freqs[0][fi] > freqs[0][secondmax]:
            thirdmax = secondmax
            secondmax = fi
        elif freqs[0][fi] > freqs[0][thirdmax]:
            thirdmax = fi
    authors = [
        "Alexandre_Dumas","Anton_Chekhov","Arthur_Conan_Doyle",
        "Benjamin_Franklin","Charles_Dickens","Franz_Kafka","Friedrich_Nietzsche",
        "Fyodor_Dostoyevsky","George_Elliot","Goethe","H_G_Wells",
        "Henry_D_Thoreau","Herman_Melville","Jack_London",
        "James_Joyce","Jane_Austen","Joseph_Conrad","Jules_Verne",
        "Leo_Tolstoy","Lewis_Carroll","Mark_Twain","Mary_Shelley",
        "Oscar_Wilde","Robert_L_Stevenson","Rudyard_Kipling","Victor_Hugo"
    ]
    return [authors[maxindex],authors[secondmax],authors[thirdmax]]
    
def predict_file(file_path, clf=None):
    """
    Params:
    file_path -> path to text file to be analysed
    clf -> GaussianNB classifier model to be used in the prediction

    Predicts the classic author a text file is closest to based on word frequency
    """
    frequencies = get_word_frequencies_file(file_path)
    if clf == None:
        clf = get_model()
    freqs,maxindex = clf.predict_log_proba([frequencies]),0
    for fi in range(26):
        if freqs[0][fi] > freqs[0][maxindex]:
            maxindex = fi
    authors = [
        "Alexandre_Dumas","Anton_Chekhov","Arthur_Conan_Doyle",
        "Benjamin_Franklin","Charles_Dickens","Franz_Kafka","Friedrich_Nietzsche",
        "Fyodor_Dostoyevsky","George_Elliot","Goethe","H_G_Wells",
        "Henry_D_Thoreau","Herman_Melville","Jack_London",
        "James_Joyce","Jane_Austen","Joseph_Conrad","Jules_Verne",
        "Leo_Tolstoy","Lewis_Carroll","Mark_Twain","Mary_Shelley",
        "Oscar_Wilde","Robert_L_Stevenson","Rudyard_Kipling","Victor_Hugo"
    ]
    return authors[maxindex]

def predict_file_top2(file_path, clf=None):
    """
    Params:
    file_path -> path to text file to be analysed
    clf -> GaussianNB classifier model to be used in the prediction

    Predicts the two classic authors a text file is closest to based on word frequency
    """
    frequencies = get_word_frequencies_file(file_path)
    if clf == None:
        clf = get_model()
    freqs,maxindex,secondmax = clf.predict_log_proba([frequencies]),0,1
    for fi in range(26):
        if freqs[0][fi] > freqs[0][maxindex]:
            maxindex = fi
            if maxindex == secondmax:
                secondmax = 0
    for fi in range(26):
        if freqs[0][fi] > freqs[0][secondmax] and fi != maxindex:
            secondmax = fi
    authors = [
        "Alexandre_Dumas","Anton_Chekhov","Arthur_Conan_Doyle",
        "Benjamin_Franklin","Charles_Dickens","Franz_Kafka","Friedrich_Nietzsche",
        "Fyodor_Dostoyevsky","George_Elliot","Goethe","H_G_Wells",
        "Henry_D_Thoreau","Herman_Melville","Jack_London",
        "James_Joyce","Jane_Austen","Joseph_Conrad","Jules_Verne",
        "Leo_Tolstoy","Lewis_Carroll","Mark_Twain","Mary_Shelley",
        "Oscar_Wilde","Robert_L_Stevenson","Rudyard_Kipling","Victor_Hugo"
    ]
    return [authors[maxindex],authors[secondmax]]

def predict_file_top3(file_path, clf=None):
    """
    Params:
    file_path -> path to text file to be analysed
    clf -> GaussianNB classifier model to be used in the prediction

    Predicts the three classic authors a text file is closest to based on word frequency
    """
    frequencies = get_word_frequencies_file(file_path)
    if clf == None:
        clf = get_model()
    freqs,maxindex,secondmax,thirdmax = clf.predict_log_proba([frequencies]),0,1,2
    for fi in range(26):
        if freqs[0][fi] > freqs[0][maxindex]:
            maxindex = fi
            if maxindex == secondmax:
                secondmax = 0
            if maxindex == thirdmax:
                if secondmax == 0:
                    thirdmax = 1
                else:
                    thirdmax = 0
    for fi in range(26):
        if freqs[0][fi] > freqs[0][secondmax] and fi != maxindex:
            secondmax = fi
            if secondmax == thirdmax:
                if 0 not in [maxindex,secondmax]:
                    thirdmax = 0
                elif 1 not in [maxindex,secondmax]:
                    thirdmax = 1
                else:
                    thirdmax = 2
    for fi in range(26):
        if freqs[0][fi] > freqs[0][thirdmax] and fi != maxindex and fi != secondmax:
            thirdmax = fi
    authors = [
        "Alexandre_Dumas","Anton_Chekhov","Arthur_Conan_Doyle",
        "Benjamin_Franklin","Charles_Dickens","Franz_Kafka","Friedrich_Nietzsche",
        "Fyodor_Dostoyevsky","George_Elliot","Goethe","H_G_Wells",
        "Henry_D_Thoreau","Herman_Melville","Jack_London",
        "James_Joyce","Jane_Austen","Joseph_Conrad","Jules_Verne",
        "Leo_Tolstoy","Lewis_Carroll","Mark_Twain","Mary_Shelley",
        "Oscar_Wilde","Robert_L_Stevenson","Rudyard_Kipling","Victor_Hugo"
    ]
    return [authors[maxindex],authors[secondmax],authors[thirdmax]]

def benchmark(c):
    """
    Params:
    c -> GaussianNB classifier to benchmark

    Measures how well the top, top2 and top3 methods do,
    prints out the results
    """
    authors = [
        "Alexandre_Dumas","Anton_Chekhov","Arthur_Conan_Doyle",
        "Benjamin_Franklin","Charles_Dickens","Franz_Kafka","Friedrich_Nietzsche",
        "Fyodor_Dostoyevsky","George_Elliot","Goethe","H_G_Wells",
        "Henry_D_Thoreau","Herman_Melville","Jack_London",
        "James_Joyce","Jane_Austen","Joseph_Conrad","Jules_Verne",
        "Leo_Tolstoy","Lewis_Carroll","Mark_Twain","Mary_Shelley",
        "Oscar_Wilde","Robert_L_Stevenson","Rudyard_Kipling","Victor_Hugo"
    ]
    count1,count2,count3, total = 0,0,0,0
    for i in range(1,20):
        
        for a in authors:
            try:
                pred = predict_file_top3(PATH_TO_CURRENT_FOLDER+"books\\"+a+"\\"+str(i)+".txt",c)
                if a != pred[0]:
                    count1+=1
                if a not in pred:
                    count2+=1
                if a not in pred:
                    count3+=1
                total+=1
            except:
                pass
    print("TOP 1 STATS:")
    print(str(count1) + " mistakes out of "+str(total))
    print("\n")
    print("TOP 2 STATS:")
    print(str(count2) + " mistakes out of "+str(total))
    print("\n")
    print("TOP 3 STATS:")
    print(str(count3) + " mistakes out of "+str(total))
    print("\n")

def generate_name(top):
    """
    Takes as input the prediction output of predict_string,predict_string_top2 or predict_string_top3
    (or the equivalents for files) and outputs a silly "name" based on which author(s) were in the prediction
    """
    name_parts = {
        "Alexandre_Dumas": ["Fourth Musketeer"," of Monte Cristo", ", wearing an Iron Mask"],
        "Anton_Chekhov": ["Uncle Vanya"," the unfired gun",", M.D."],
        "Arthur_Conan_Doyle": ["Sidekick Watson"," the consulting detective",", amateur detective"],
        "Benjamin_Franklin": ["Founding Father"," the polymath",", a.k.a Poor Rick"],
        "Charles_Dickens": ["Mr Scrooge"," the not-magical-Copperfield",", full of expectations"],
        "Franz_Kafka": ["K"," Kafkaesque",", already half-bug"],
        "Friedrich_Nietzsche": ["Antichrist"," the Dead God",", a gay scientist"],
        "Fyodor_Dostoyevsky": ["Idiot"," the Punished",", writing from Underground"],
        "George_Elliot": ["Romola"," marching through the Middle",", a genuine Victorian"],
        "Goethe": ["Mephistopheles"," Wolfgang",", full of sorrow"],
        "H_G_Wells": ["Invisible Man"," the First Moon Man",", at war with Mars"],
        "Henry_D_Thoreau": ["Wald-man"," the Walk-man",", disobedient but civil"],
        "Herman_Melville": ["Moby-Dick"," the Whale Hunter",", fan of big-game fishing"],
        "Jack_London": ["White Fang"," the Sea-Wolf",", calling the wild"],
        "James_Joyce": ["Dubliner"," the portrait artist",", also known as Odysseus"],
        "Jane_Austen": ["Proud Prejudicer"," the Proud",", sensitive and sensible"],
        "Joseph_Conrad": ["Lord Jim"," the Western-eyed",", with a dark heart"],
        "Jules_Verne": ["15 and Captain"," the World-Traveller",", currently deep under the sea"],
        "Leo_Tolstoy": ["Anna Karenina"," from an unhappy family",", with a really cool beard"],
        "Lewis_Carroll": ["Alice"," the Red Queen",", way down the Rabbit Hole"],
        "Mark_Twain": ["Tom S."," the Pauper Prince",", off having Adventures"],
        "Mary_Shelley": ["Frankenstein"," the Last Man",", BFFs with Byron"],
        "Oscar_Wilde": ["Dorian"," the Selfish Giant",", with a painting for a soul"],
        "Robert_L_Stevenson": ["Treasurer of Islands"," and Mr Hyde",", travelling with a donkey"],
        "Rudyard_Kipling": ["Mowgli"," the Indian",", author of just so literature"],
        "Victor_Hugo": ["Les Miserable"," the Hunchback",", with a very masculine laugh"]
    }

    # input is a string, only want the top outcome expressed
    if len(top) > 3:
        return name_parts[top][0]
    elif len(top) == 2:
        return name_parts[top[0]][0] + name_parts[top[1]][1]
    else:
        return name_parts[top[0]][0] + name_parts[top[1]][1] + name_parts[top[2]][2]