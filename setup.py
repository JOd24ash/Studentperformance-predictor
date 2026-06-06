from setuptools import find_packages,setup
from typing import List

hypene_dot='-e .'

def get_requirnment(file_path:str)->List[str]:
    #this is the code will auto mat ethe install requires
    requiremwnts=[]
    with open (file_path) as file_obj:
        requiremwnts=file_obj.readlines()
        requiremwnts=[req.replace('\n','') for  req in requiremwnts]

        if hypene_dot in requiremwnts:
            requiremwnts.remove(hypene_dot)
    return requiremwnts        

setup(
    name="mlproject",
    version="0.0.1",
    author="JOD",
    author_email="tsxmiky24@gamil.com",
    packages=find_packages(),
    install_requires=get_requirnment("requirement.txt"),
)