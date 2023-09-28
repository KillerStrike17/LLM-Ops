import os 
from typing import List

class TextFileLoader:
    """
    This class has the functionality to load the data from 
    the text files.
    """
    def __init__(self, path:str, encoding:str = "utf-8")->None:
        self.documents = []
        self.path = path
        self.encoding = encoding
    
    def load(self)->None:
        """
        if the path is of a directory, then load directory and read the file,
        else if the path is of the file, directly read the file.
        """
        if os.path.isdir(self.path):
            self.load_directory()
        elif os.path.isfile(self.path) and self.path.endswith(".txt"):
            self.load_file()
        else:
            raise ValueError(
                "Provided path is neither a valid directory not a .txt tile"
            )
    
    def load_file(self)->None:
        """
        read the text file and append it to the list
        """
        with open(self.path,"r",encoding=self.encoding) as f:
            self.documents.append(f.read())
    
    def load_directory(self)->None:
        """
        reads all the text files in the directory and appends it to the list
        """
        for root, _, files in os.walk(self.path):
            for file in files:
                if file.endswith(".txt"):
                    with open(
                        os.path.join(root, file),"r",encoding=self.encoding
                    ) as f:
                        self.documents.append(f.read())
    
    def load_documents(self):
        """
        call the load function, that calls the function to read data and returns the documents.
        """
        self.load()
        return self.documents

class CharacterTextSplitter:
    """
    This class contains the functionailites to chunk the text documents.
    """
    def __init__(self, chunk_size:int = 1000,chunk_overlap:int = 200):
        assert(chunk_size>chunk_overlap),"Chunk size must be greater than chunk overlap"
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text:str)->List[str]:
        """
        takes in text and splits them based on character count
        """
        chunks = []
        for i in range(0, len(text),self.chunk_size-self.chunk_overlap):
            chunks.append(text[i:i+self.chunk_size])
        return chunks
    
    def split_texts(self, texts:List[str])->List[str]:
        """
        takes in list of texts and breaks it down to chunks
        """
        chunks = []
        for text in texts:
            chunks.extend(self.split(text))
        return chunks
    
if __name__ == "__main__":
    loader = TextFileLoader("/Users/shubham.agnihotri/Documents/GitHub/LLM-Ops/RAQA from scratch/data/KingLear.txt")
    loader.load()
    splitter = CharacterTextSplitter()
    chunks = splitter.split_texts(loader.documents)
    print(len(chunks))
    print(chunks[0])
    print("--------")
    print(chunks[1])
    print("--------")
    print(chunks[-2])
    print("--------")
    print(chunks[-1])
    

