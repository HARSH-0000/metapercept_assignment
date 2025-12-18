

from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, FOAF
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import numpy as np

class KnowledgeGraphSystem:
    def __init__(self):
        
        self.graph = Graph()
        self.ns = Namespace("http://example.org/")
        self.sentences = []
        self.triple_map = {}
        
       
        print("Loading embedding model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
       
        print("Initializing vector database...")
        self.client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            allow_reset=True
        ))
        self.collection = None
        
    def build_rdf_graph(self):
       
        print("\n=== Building RDF Knowledge Graph ===")
        
       
        alice = URIRef(self.ns.Alice)
        bob = URIRef(self.ns.Bob)
        charlie = URIRef(self.ns.Charlie)
        acme = URIRef(self.ns.AcmeCorp)
        techstart = URIRef(self.ns.TechStart)
        
       
        self.graph.add((alice, RDF.type, self.ns.Person))
        self.graph.add((bob, RDF.type, self.ns.Person))
        self.graph.add((charlie, RDF.type, self.ns.Person))
        self.graph.add((acme, RDF.type, self.ns.Company))
        self.graph.add((techstart, RDF.type, self.ns.Company))
        
       
        self.graph.add((alice, self.ns.knows, bob))
        self.graph.add((alice, self.ns.knows, charlie))
        self.graph.add((bob, self.ns.knows, alice))
        
      
        self.graph.add((bob, self.ns.worksAt, acme))
        self.graph.add((charlie, self.ns.worksAt, techstart))
        self.graph.add((alice, self.ns.worksAt, acme))
        
        
        self.graph.add((acme, self.ns.locatedIn, Literal("New York")))
        self.graph.add((techstart, self.ns.locatedIn, Literal("San Francisco")))
        
        
        self.graph.add((alice, self.ns.age, Literal(30)))
        self.graph.add((bob, self.ns.age, Literal(28)))
        self.graph.add((charlie, self.ns.age, Literal(32)))
        
        print(f"✓ Added {len(self.graph)} RDF triples to the graph")
        
    def convert_triples_to_sentences(self):
      
        print("\n=== Converting Triples to Sentences ===")
        
        self.sentences = []
        self.triple_map = {}
        
        for idx, (subj, pred, obj) in enumerate(self.graph):
           
            subj_name = str(subj).split('#')[-1].split('/')[-1]
            pred_name = str(pred).split('#')[-1].split('/')[-1]
            obj_name = str(obj).split('#')[-1].split('/')[-1] if isinstance(obj, URIRef) else str(obj)
            
      
            if pred_name == "type":
                sentence = f"{subj_name} is a {obj_name}"
            elif pred_name == "knows":
                sentence = f"{subj_name} knows {obj_name}"
            elif pred_name == "worksAt":
                sentence = f"{subj_name} works at {obj_name}"
            elif pred_name == "locatedIn":
                sentence = f"{subj_name} is located in {obj_name}"
            elif pred_name == "age":
                sentence = f"{subj_name} is {obj_name} years old"
            else:
                sentence = f"{subj_name} {pred_name} {obj_name}"
            
            self.sentences.append(sentence)
            self.triple_map[idx] = (subj_name, pred_name, obj_name)
            print(f"  {idx+1}. {sentence}")
        
        print(f"✓ Created {len(self.sentences)} sentences from RDF triples")
        
    def create_embeddings_and_store(self):
       
        print("\n=== Creating Embeddings and Storing in ChromaDB ===")
        
        
        try:
            self.client.delete_collection("knowledge_graph")
        except:
            pass
        
        self.collection = self.client.create_collection(
            name="knowledge_graph",
            metadata={"description": "RDF knowledge graph sentences"}
        )
        
       
        print("Generating embeddings...")
        embeddings = self.model.encode(self.sentences, show_progress_bar=True)
        
        
        print("Storing in vector database...")
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=self.sentences,
            ids=[f"triple_{i}" for i in range(len(self.sentences))],
            metadatas=[{"triple": f"{self.triple_map[i]}"} for i in range(len(self.sentences))]
        )
        
        print(f"✓ Stored {len(self.sentences)} embeddings in ChromaDB")
        
    def query_knowledge_graph(self, query, top_k=3):
        
        print(f"\n=== Query: '{query}' ===")
        
       
        query_embedding = self.model.encode([query])[0]
        
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        print(f"\nTop {top_k} Results:")
        print("-" * 70)
        
        for idx, (doc, distance, metadata) in enumerate(zip(
            results['documents'][0],
            results['distances'][0],
            results['metadatas'][0]
        ), 1):
            similarity = 1 - distance  
            print(f"   Similarity Score: {similarity:.4f}")
            print(f"   RDF Triple: {metadata['triple']}")
        
        return results
    
    


def main():
   
    print("="*70)
    print("RDF KNOWLEDGE GRAPH WITH SEMANTIC SEARCH")
    print("="*70)
    
    
    kg_system = KnowledgeGraphSystem()
    
   
    kg_system.build_rdf_graph()
    
    kg_system.convert_triples_to_sentences()
    
   
    kg_system.create_embeddings_and_store()
    
    
    kg_system.display_graph_stats()
    
    
    print("\n" + "="*70)
    print("RUNNING EXAMPLE QUERIES")
    print("="*70)
    
    queries = [
        "Who does Alice know?",
        "Where does Bob work?",
        "Who works at Acme Corp?",
        "How old is Charlie?",
        "Which companies are in the graph?"
    ]
    
    for query in queries:
        kg_system.query_knowledge_graph(query, top_k=3)
        print()
    
  
    print("\n" + "="*70)
    print("INTERACTIVE QUERY MODE")
    print("="*70)
    print("Enter your queries (or 'quit' to exit):\n")
    
    while True:
        user_query = input("Query: ").strip()
        
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("\nThank you for using the Knowledge Graph System!")
            break
        
        if user_query:
            kg_system.query_knowledge_graph(user_query, top_k=3)
        else:
            print("Please enter a valid query.")


if __name__ == "__main__":
    main()
