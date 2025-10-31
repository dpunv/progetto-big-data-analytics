"""
Modulo per scaricare e processare articoli Wikipedia per generare embeddings reali.

SCOPO:
- Scaricare articoli Wikipedia in italiano/inglese
- Estrarre frasi significative
- Generare embeddings usando sentence-transformers
- Creare dataset realistico per clustering semantico

VANTAGGI DATI REALI:
- Distanze semantiche più variegate (0.1-4.0 invece di 0.83-1.01)
- Cluster più distintivi (tech, sport, geografia, arte, etc.)
- Test realistici del sistema di routing semantico
"""

import wikipedia
import numpy as np
from sentence_transformers import SentenceTransformer
import re
from tqdm import tqdm
import joblib
import os
from typing import List, Tuple

class WikipediaEmbeddingGenerator:
    """
    Genera embeddings da articoli Wikipedia.
    
    PROCESSO:
    1. Scarica articoli Wikipedia da categorie diverse
    2. Estrae frasi pulite (no liste, no codice, no riferimenti)
    3. Genera embeddings con modello sentence-transformer
    4. Salva dataset per riutilizzo
    
    CATEGORIE WIKIPEDIA COPERTE:
    - Technology (AI, ML, programming, etc.)
    - Sports (football, basketball, tennis, etc.)
    - Geography (countries, cities, landmarks, etc.)
    - Science (physics, chemistry, biology, etc.)
    - Arts (painting, music, literature, etc.)
    - History (wars, civilizations, events, etc.)
    
    Questo crea cluster semanticamente MOLTO distinti!
    """
    
    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2', language: str = 'en'):
        """
        Inizializza il generatore.
        
        Args:
            model_name: Nome del modello sentence-transformers
                - 'paraphrase-multilingual-MiniLM-L12-v2': 384-dim, supporta italiano/inglese
                - 'all-MiniLM-L6-v2': 384-dim, solo inglese, più veloce
                - 'all-mpnet-base-v2': 768-dim, migliore qualità, più lento
            language: Lingua Wikipedia ('en', 'it', 'es', etc.)
        """
        print(f"Loading sentence-transformer model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        wikipedia.set_lang(language)
        self.language = language
        
        print(f"✓ Model loaded: {self.embedding_dim}-dimensional embeddings")
        print(f"✓ Wikipedia language: {language}")
    
    def get_diverse_topics(self) -> dict[str, List[str]]:
        """
        Restituisce topics Wikipedia divisi per categorie semantiche.
        
        STRATEGIA:
        - Scelgo ~50 articoli da 6 categorie diverse
        - Ogni categoria ha temi distinti → cluster semanticamente separati
        - Questo garantisce range di distanze ampio (0.3-4.0)
        
        Returns:
            Dizionario {categoria: [lista_articoli]}
        """
        if self.language == 'en':
            return {
                'technology': [
                    'Artificial intelligence', 'Machine learning', 'Deep learning', 
                    'Neural network', 'Python (programming language)', 'JavaScript',
                    'Cloud computing', 'Blockchain', 'Cryptocurrency', 'Quantum computing'
                ],
                'sports': [
                    'Association football', 'Basketball', 'Tennis', 'Cricket',
                    'Formula One', 'Olympics', 'FIFA World Cup', 'NBA', 'UEFA Champions League',
                    'Marathon', 'Swimming (sport)', 'Boxing'
                ],
                'geography': [
                    'Italy', 'United States', 'China', 'Brazil', 'Australia',
                    'Mount Everest', 'Amazon River', 'Sahara', 'Pacific Ocean',
                    'Rome', 'New York City', 'Tokyo', 'Paris'
                ],
                'science': [
                    'Physics', 'Chemistry', 'Biology', 'DNA', 'Evolution',
                    'Photosynthesis', 'Black hole', 'Quantum mechanics', 'Theory of relativity',
                    'Periodic table', 'Cell (biology)', 'Atom'
                ],
                'arts': [
                    'Leonardo da Vinci', 'Vincent van Gogh', 'Pablo Picasso',
                    'Music', 'Classical music', 'Jazz', 'Rock music',
                    'Literature', 'Poetry', 'William Shakespeare', 'Cinema'
                ],
                'history': [
                    'World War II', 'Ancient Rome', 'Ancient Egypt', 'Renaissance',
                    'Industrial Revolution', 'French Revolution', 'American Revolution',
                    'Middle Ages', 'Cold War', 'Vikings'
                ]
            }
        elif self.language == 'it':
            return {
                'tecnologia': [
                    'Intelligenza artificiale', 'Apprendimento automatico', 'Python',
                    'JavaScript', 'Cloud computing', 'Blockchain', 'Informatica',
                    'Internet', 'Computer', 'Smartphone'
                ],
                'sport': [
                    'Calcio', 'Pallacanestro', 'Tennis', 'Formula 1',
                    'Olimpiadi', 'Coppa del Mondo FIFA', 'Serie A', 'NBA',
                    'Ciclismo', 'Nuoto', 'Atletica leggera'
                ],
                'geografia': [
                    'Italia', 'Stati Uniti', 'Cina', 'Brasile', 'Australia',
                    'Roma', 'Milano', 'Napoli', 'Firenze', 'Venezia',
                    'Alpi', 'Mediterraneo', 'Po (fiume)'
                ],
                'scienza': [
                    'Fisica', 'Chimica', 'Biologia', 'DNA', 'Evoluzione',
                    'Fotosintesi', 'Buco nero', 'Meccanica quantistica',
                    'Tavola periodica', 'Cellula', 'Atomo'
                ],
                'arte': [
                    'Leonardo da Vinci', 'Michelangelo', 'Raffaello',
                    'Musica', 'Musica classica', 'Opera lirica',
                    'Letteratura italiana', 'Dante Alighieri', 'Cinema'
                ],
                'storia': [
                    'Seconda guerra mondiale', 'Impero romano', 'Rinascimento',
                    'Risorgimento', 'Medioevo', 'Rivoluzione francese',
                    'Antica Roma', 'Repubblica di Venezia'
                ]
            }
        else:
            # Fallback inglese
            return self.get_diverse_topics_for_language('en')
    
    def extract_sentences(self, text: str, max_sentences: int = 50) -> List[str]:
        """
        Estrae frasi pulite da testo Wikipedia.
        
        COSA RIMUOVE:
        - Riferimenti [1], [2], etc.
        - Parentesi con date/info extra
        - Liste puntate
        - Frasi troppo corte (< 20 caratteri)
        - Frasi troppo lunghe (> 500 caratteri)
        
        Args:
            text: Testo grezzo Wikipedia
            max_sentences: Massimo numero di frasi da estrarre
            
        Returns:
            Lista di frasi pulite
        """
        # Rimuovi riferimenti [1], [citation needed], etc.
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'\[citation needed\]', '', text)
        text = re.sub(r'\[.*?\]', '', text)
        
        # Rimuovi sezioni non utili
        text = re.sub(r'==.*?==', '', text)
        
        # Split in frasi (semplice: split su . ! ?)
        sentences = re.split(r'[.!?]+', text)
        
        # Pulisci e filtra
        clean_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            
            # Salta frasi troppo corte o troppo lunghe
            if len(sentence) < 20 or len(sentence) > 500:
                continue
            
            # Salta se inizia con caratteri strani (liste, etc.)
            if sentence.startswith(('-', '*', '•', '=')):
                continue
            
            # Salta se troppi caratteri speciali (probabilmente tabelle/codice)
            special_chars = sum(1 for c in sentence if not c.isalnum() and c not in ' .,!?-\'\"')
            if special_chars > len(sentence) * 0.3:
                continue
            
            clean_sentences.append(sentence)
            
            if len(clean_sentences) >= max_sentences:
                break
        
        return clean_sentences
    
    def download_and_embed(self, n_samples: int = 100000, cache_path: str = 'wikipedia_embeddings_cache.pkl') -> Tuple[np.ndarray, List[str]]:
        """
        Scarica articoli Wikipedia e genera embeddings.
        
        PROCESSO:
        1. Per ogni categoria: scarica articoli
        2. Estrai ~50 frasi per articolo
        3. Genera embeddings con sentence-transformer
        4. Salva cache per riutilizzo
        
        VANTAGGI CACHE:
        - Download Wikipedia è lento (~5-10 min per 50 articoli)
        - Encoding è lento (~2-3 min per 10K frasi)
        - Cache permette di riutilizzare stesso dataset
        
        Args:
            n_samples: Numero target di frasi (distribuito tra categorie)
            cache_path: Percorso cache
            
        Returns:
            (embeddings, sentences): Array embeddings + lista frasi originali
        """
        # Controlla cache
        if os.path.exists(cache_path):
            print(f"Loading cached Wikipedia embeddings from {cache_path}...")
            data = joblib.load(cache_path)
            embeddings = data['embeddings']
            sentences = data['sentences']
            print(f"✓ Loaded {len(embeddings)} cached embeddings")
            
            # Se cache ha meno samples del richiesto, avvisa
            if len(embeddings) < n_samples:
                print(f"⚠️  Cache ha solo {len(embeddings)} embeddings, ma ne sono richiesti {n_samples}")
                print(f"    Ricampiono i dati disponibili...")
                # Ricampia con replacement per raggiungere il target
                indices = np.random.choice(len(embeddings), n_samples, replace=True)
                embeddings = embeddings[indices]
                sentences = [sentences[i] for i in indices]
            elif len(embeddings) > n_samples:
                # Taglia se cache ha più del necessario
                embeddings = embeddings[:n_samples]
                sentences = sentences[:n_samples]
            
            return embeddings, sentences
        
        print(f"\n{'='*70}")
        print(f"DOWNLOADING WIKIPEDIA ARTICLES")
        print(f"{'='*70}")
        print(f"Target: {n_samples} sentence embeddings")
        print(f"Language: {self.language}")
        print(f"Embedding model: {self.model.get_sentence_embedding_dimension()}-dim\n")
        
        topics_by_category = self.get_diverse_topics()
        all_sentences = []
        category_stats = {}
        
        # PRIMO PASS: scarica tutti gli articoli disponibili
        print("📥 FIRST PASS: Scaricamento massivo da tutte le categorie\n")
        
        for category, topics in topics_by_category.items():
            print(f"📚 Category: {category.upper()}")
            
            category_sentences = []
            articles_downloaded = 0
            
            for topic in tqdm(topics, desc=f"   Downloading {category}", leave=False):
                try:
                    page = wikipedia.page(topic, auto_suggest=False)
                    sentences = self.extract_sentences(page.content, max_sentences=100)  # Aumenta da 50 a 100
                    category_sentences.extend(sentences)
                    articles_downloaded += 1
                
                except (wikipedia.exceptions.DisambiguationError, 
                        wikipedia.exceptions.PageError,
                        Exception):
                    continue
            
            all_sentences.extend(category_sentences)
            category_stats[category] = {
                'sentences': len(category_sentences),
                'articles': articles_downloaded
            }
            
            print(f"   ✓ {articles_downloaded} articles → {len(category_sentences)} sentences")
        
        print(f"\n{'='*70}")
        print(f"DOWNLOAD SUMMARY")
        print(f"{'='*70}")
        total_collected = 0
        for category, stats in category_stats.items():
            print(f"  {category:15s}: {stats['articles']:3d} articles → {stats['sentences']:6d} sentences")
            total_collected += stats['sentences']
        print(f"  {'TOTAL':15s}: {total_collected:6d} sentences")
        
        # Verifica se abbiamo abbastanza frasi
        if total_collected < n_samples:
            print(f"\n⚠️  ATTENZIONE: Raccolte solo {total_collected} frasi, ma ne servono {n_samples}")
            print(f"   Ricampiono con replacement per raggiungere il target...\n")
            
            # Ricampia con replacement per raggiungere il target
            indices = np.random.choice(len(all_sentences), n_samples - total_collected, replace=True)
            resampled = [all_sentences[i] for i in indices]
            all_sentences.extend(resampled)
            
            print(f"   ✓ Aggiunte {len(resampled)} frasi ricampionate")
            print(f"   ✓ Totale finale: {len(all_sentences)} frasi\n")
        
        # Taglia se abbiamo troppo
        if len(all_sentences) > n_samples:
            all_sentences = all_sentences[:n_samples]
        
        # Genera embeddings
        print(f"🔄 Generating embeddings...")
        embeddings = self.model.encode(
            all_sentences, 
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True
        )
        
        print(f"✓ Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
        
        # Salva cache
        print(f"\n💾 Saving cache to {cache_path}...")
        joblib.dump({
            'embeddings': embeddings,
            'sentences': all_sentences,
            'category_stats': category_stats,
            'model_name': self.model.get_sentence_embedding_dimension(),
            'language': self.language,
            'total_target': n_samples
        }, cache_path)
        
        print(f"✓ Cache saved successfully!")
        print(f"\n{'='*70}\n")
        
        return embeddings, all_sentences


def load_wikipedia_embeddings(n_samples: int = 10000, 
                              model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2',
                              language: str = 'en',
                              cache_path: str = 'wikipedia_embeddings_cache.pkl') -> np.ndarray:
    """
    Funzione helper per caricare embeddings Wikipedia (con cache).
    
    USO SEMPLICE:
    ```python
    embeddings = load_wikipedia_embeddings(n_samples=10000)
    ```
    
    Args:
        n_samples: Numero di embeddings desiderati
        model_name: Modello sentence-transformer
        language: Lingua Wikipedia
        cache_path: Path cache
        
    Returns:
        Array numpy [n_samples, embedding_dim]
    """
    generator = WikipediaEmbeddingGenerator(model_name=model_name, language=language)
    embeddings, _ = generator.download_and_embed(n_samples=n_samples, cache_path=cache_path)
    
    # Se cache ha più/meno samples, taglia/ripeti
    if len(embeddings) > n_samples:
        embeddings = embeddings[:n_samples]
    elif len(embeddings) < n_samples:
        # Ripeti embeddings se necessario (unlikely)
        n_repeats = (n_samples // len(embeddings)) + 1
        embeddings = np.tile(embeddings, (n_repeats, 1))[:n_samples]
    
    return embeddings.astype('float32')
