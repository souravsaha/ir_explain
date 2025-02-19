from setuptools import setup, find_packages

setup(
    name='ir_explain',  # Replace with your project name
    version='0.1',
    description='A Python Library for Explainable IR methods',
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author='Sourav Saha',
    author_email='souravsaha.juit@gmail.com',
    url='https://github.com/souravsaha/ir_explain/',  # Replace with your project's URL
    packages=find_packages(),  # Automatically find packages in your project
    install_requires=[
        'ipython==8.12.3',
        'matplotlib==3.9.4',
        'nltk==3.9.1',
        #'numpy==2.0.2',
        'numpy==1.24.4',
        'pandas==2.2.3',
        'rank_bm25==0.2.2',
        'scikit_learn==1.6.0',
        'scipy==1.10.1',
        'sentence_transformers==3.3.1',
        'scikit-image',
        'torch==2.2.0',
        'ir_datasets',
        'tqdm==4.67.1',
        'pyserini==0.21.0',
        'gensim==4.3.1',
        'torchtext==0.17.0',
        'h5py',
        'captum==0.7.0',
        'genosolver==0.1.0.6',
        'cvxpy==1.3.2',
        'pytorch-lightning',
        'faiss-cpu==1.8.0'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific :: Information Retrieval',
    ],
    python_requires='>=3.9',  # Specify the Python versions compatible with your project
)
