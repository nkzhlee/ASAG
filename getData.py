import xml4h
import os
import re
import spacy
#doc = xml4h.parse('tests/data/monty_python_films.xml')
#https://xml4h.readthedocs.io/en/latest/


class getData:
    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.files = []

        # test
        self.testfile = None
        self.text_questions_dict = {}
        self.text_answers_dict = {}

    def getXML(self):
        for path, dirnames, filenames in os.walk(self.path):
            # print('{} {} {}'.format(repr(path), repr(dirnames), repr(filenames)))
            for file in filenames:
                if os.path.splitext(file)[1] == '.xml':
                    file_path = path + "/" + file
                    self.files.append(file_path)
        for file in self.files:
            print(file)

    def test(self):
        # get data

        doc = xml4h.parse('/Users/zhaohuilee/Desktop/RA/2020-fall/ASGA/data/semeval2013-Task7-2and3way/training/3way/beetle/FaultFinding-BULB_C_VOLTAGE_EXPLAIN_WHY1.xml')
        q_id = print(doc.question["id"])
        q_text = print(doc.question.questionText.text)
        self.text_questions_dict[q_id] = q_text
        for st_ans in doc.question.studentAnswers.studentAnswer[:3]:
            print(st_ans.text)



    def tokenize(text):
        tok = spacy.load('en')
        text = re.sub(r"[^\x00-\x7F]+", " ", text)
        regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')  # remove punctuation and numbers
        nopunct = regex.sub(" ", text.lower())
        return [token.text for token in tok.tokenizer(nopunct)]

    def return3way(self):
        print("Hello my name is " + self.name)



if __name__ == "__main__":
    print("start ")
    three_way = getData("3-way",
    "/Users/zhaohuilee/Desktop/RA/2020-fall/ASGA/data/semeval2013-Task7-2and3way/training/3way/beetle")
    ## three_way.getXML()
    three_way.test()
    print("done")