import re
#import spacy
#from fuzzywuzzy import process
from typing import List, Dict, Optional
import pandas as pd

class HybridParser:
    def __init__(self):
        #self.nlp = spacy.load("pl_core_news_sm")
        self.medical_patterns = {
            # Nazwy leków (małe litery)
            'drug_name': r'^[a-ząćęłńóśźż][a-ząćęłńóśźż\s]+$',
            
            # Dawkowanie
            'dosage': r'(\d+(?:\.\d+)?)\s*(mg|g|ml|mcg|j\.m\.)(?:/(?:kg|m2))?',
            
            # Częstotliwość
            'frequency': [
                r'co (\d+) (tygodn\w+|dni\w*)',
                r'(\d+)x\s+(?:na dobę|dziennie|w tygodniu)',
                r'raz (?:na|w) (\d+) tygodni',
            ],
            
            # Czas trwania
            'duration': r'przez (\d+) (?:kolejnych )?(?:dni|tygodni|miesięcy)',
            
            # Maksymalne opóźnienie
            'max_delay': r'maksymalny czas opóźnienia.*?(\d+) tygodni',
            
            # Cykl
            'cycle': r'(\d+)-dniowy(?:ego)? cykl',
            
            # Skojarzenie
            'combination': r'w skojarzeniu z (\w+)',
            
            # Powierzchnia ciała
            'body_surface': r'(\d+(?:\.\d+)?)\s*mg/m2',
            
            # Masa ciała
            'body_weight': r'(\d+(?:\.\d+)?)\s*mg/kg',
        }
        #self.llm_client = OpenAI() if USE_LLM else None

    def try_regex(self, text, column_number):
        if column_number==1:
            return self.pragmatic_parse_program(text)
        elif column_number==2:
            return self.pragmatic_parse_dosage(text)
        elif column_number==3:
            return self.pragmatic_parse_monitoring(text)
        return None
    
    def parse(self, cell_text, column_number):
        """Multi-strategy parsing"""
        
        # Quick win: regex dla prostych przypadków (szybkie)
        if result := self.try_regex(cell_text, column_number):
            return {"data": result, "method": "regex", "confidence": 0.95}
        
        # Medium effort: fuzzy matching (średnie)
        if result := self.try_fuzzy(cell_text,  column_number):
            return {"data": result, "method": "fuzzy", "confidence": 0.85}
        
        # More effort: NER (wolniejsze ale dokładniejsze)
        if result := self.try_ner(cell_text, column_number):
            return {"data": result, "method": "ner", "confidence": 0.80}
        
        # Last resort: LLM (najwolniejsze, najdroższe)
        # if self.llm_client:
        #     result = self.try_llm(cell_text)
        #     return {"data": result, "method": "llm", "confidence": 0.75}
        
        # Fallback: raw text do manual review
        return {
            "data": {"column":column_number,"raw": cell_text}, 
            "method": "none", 
            "confidence": 0.0,
            "needs_review": True
        }
    
    
    
    def try_fuzzy(self, text, column_number):
        # """Attempt to parse using fuzzy matching"""
        # choices = self.known_entities.get(column_number, [])
        # best_match, score = process.extractOne(text, choices)
        # if score > 80:
        #     return {"matched_entity": best_match}
        return None
    
    def try_ner(self, text, column_number):
        return None
    
    def pragmatic_parse_dosage(self,text: str) -> List[Dict]:
        """Pragmatyczne parsowanie dla POC - łączy regex z boundary detection"""
        
        result = {
            'section_title': None,
            'drugs': [],
            'general_modifications': None
        }
        
        # 1. Wyciągnij tytuł sekcji (pierwsza linia)
        lines = text.strip().split('\n')
        result['section_title'] = lines[0] if lines else None
        
        # 2. Znajdź wszystkie nazwy leków (małe litery na początku linii)
        drug_names = []
        for i, line in enumerate(lines):
            # Lek to: mała litera na początku + następna linia zaczyna się od "Zalecana" lub "Leczenie"
            if (line and line[0].islower() and 
                i + 1 < len(lines) and 
                (lines[i+1].startswith('Zalecana') or lines[i+1].startswith('Leczenie'))):
                drug_names.append((i, line.strip()))
        
        # 3. Podziel tekst na sekcje leków
        for idx, (line_num, drug_name) in enumerate(drug_names):
            # Znajdź początek i koniec sekcji leku
            start_idx = line_num
            
            # Koniec to: początek następnego leku lub sekcja "Modyfikacje"
            if idx + 1 < len(drug_names):
                end_idx = drug_names[idx + 1][0]
            else:
                # Szukaj "Modyfikacje dawkowania"
                mod_idx = next((i for i, l in enumerate(lines) 
                            if 'Modyfikacje dawkowania' in l), len(lines))
                end_idx = mod_idx
            
            # Wyciągnij tekst sekcji leku
            drug_section = '\n'.join(lines[start_idx:end_idx])
            
            # Parsuj szczegóły leku
            drug_data = self.parse_single_drug(drug_name, drug_section)
            result['drugs'].append(drug_data)
        
        # 4. Wyciągnij sekcję "Modyfikacje dawkowania"
        mod_idx = next((i for i, l in enumerate(lines) 
                    if 'Modyfikacje dawkowania' in l), None)
        if mod_idx:
            result['general_modifications'] = '\n'.join(lines[mod_idx:])
        
        return result

    def parse_single_drug(self, drug_name: str, section_text: str) -> Dict:
        """
        Parsuje sekcję pojedynczego leku
        """
        
        drug_data = {
            'name': drug_name,
            'raw_text': section_text,
            'dosage_info': {},
            'cycle_info': {},
            'delay_info': {},
            'combination_info': {},
            'special_instructions': []
        }
        
        # Wyciągnij kluczowe informacje
        
        # Dawkowanie
        dose_matches = re.finditer(
            r'(?:zalecana dawka|dawka|dawkowanie).*?(\d+(?:\.\d+)?)\s*(mg|g|ml|mcg)(?:/kg)?',
            section_text,
            re.IGNORECASE
        )
        for match in dose_matches:
            dose = f"{match.group(1)} {match.group(2)}"
            if dose not in drug_data['dosage_info'].values():
                key = f"dose_{len(drug_data['dosage_info']) + 1}"
                drug_data['dosage_info'][key] = dose
        
        # Częstotliwość
        freq_matches = re.finditer(
            r'co (\d+) (tygodn\w+|dni)',
            section_text,
            re.IGNORECASE
        )
        for match in freq_matches:
            freq = f"co {match.group(1)} {match.group(2)}"
            key = f"frequency_{len(drug_data['cycle_info']) + 1}"
            drug_data['cycle_info'][key] = freq
        
        # Maksymalne opóźnienie
        delay_match = re.search(
            r'maksymalny czas opóźnienia.*?(\d+) tygodni',
            section_text
        )
        if delay_match:
            drug_data['delay_info']['max_weeks'] = int(delay_match.group(1))
        
        # Skojarzenie z innym lekiem
        combo_match = re.search(
            r'w skojarzeniu z (\w+)',
            section_text
        )
        if combo_match:
            drug_data['combination_info']['partner_drug'] = combo_match.group(1)
        
        # Fazy leczenia
        if 'faza indukująca' in section_text.lower():
            drug_data['special_instructions'].append('has_induction_phase')
        if 'faza podtrzymująca' in section_text.lower():
            drug_data['special_instructions'].append('has_maintenance_phase')
        
        # Zdania (dla dalszej analizy)
        sentences = re.split(r'(?<=[.!?])\s+', section_text)
        drug_data['sentences'] = [s.strip() for s in sentences if s.strip()]
        
        return drug_data
    
    def pragmatic_parse_monitoring(self, text: str) -> Dict:
        """
        Pragmatyczne parsowanie sekcji badań i monitorowania
        Struktura: sekcje główne -> listy badań -> warunki specyficzne dla leków
        """
        
        result = {
            'sections': [],
            'full_structure': {}
        }
        
        # Identyfikuj główne sekcje (wielkie litery na początku, brak wcięcia)
        section_pattern = r'^([A-ZĄĆĘŁŃÓŚŹŻ][^\n]+)$'
        
        if isinstance(text, str):
            lines = text.strip().split('\n')
        elif isinstance(text, list):
            # Handle the case where text is a list
            # Optionally join the list into a string before splitting
            text = ' '.join(text)
            lines = text.strip().split('\n')
        else:
            # Handle other unexpected cases
            raise ValueError("Expected a string or list, got {type(text)}")
        current_section = None
        current_items = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Czy to header sekcji? (zaczyna się wielką literą, nie ma średnika na końcu)
            if re.match(section_pattern, line) and not line.endswith(';'):
                # Zapisz poprzednią sekcję
                if current_section:
                    result['sections'].append({
                        'title': current_section,
                        'items': current_items.copy(),
                        'parsed_items': parse_test_items(current_items)
                    })
                
                # Nowa sekcja
                current_section = line
                current_items = []
            
            # Czy to item listy? (zaczyna się małą literą lub jest wcięty, kończy się średnikiem)
            elif line[0].islower() or line.endswith(';') or line.endswith('.'):
                current_items.append(line.rstrip(';').rstrip('.'))
        
        # Dodaj ostatnią sekcję
        if current_section:
            result['sections'].append({
                'title': current_section,
                'items': current_items.copy(),
                'parsed_items': parse_test_items(current_items)
            })
        
        # Stwórz strukturę hierarchiczną
        result['full_structure'] = self.create_monitoring_structure(result['sections'])
        
        return result

    def parse_test_items(items: List[str]) -> List[Dict]:
        """
        Parsuje pojedyncze pozycje testów/badań
        Wyciąga: nazwę badania, warunki, leki których dotyczy
        """
        
        parsed = []
        
        for item in items:
            test_data = {
                'raw_text': item,
                'test_name': None,
                'conditions': [],
                'specific_drugs': [],
                'frequency': None,
                'type': classify_test_type(item)
            }
            
            # Wyciągnij nazwę testu (część przed "–" lub całość jeśli nie ma)
            if '–' in item or '—' in item or ' - ' in item:
                # Podziel na nazwę testu i warunki
                parts = re.split(r'[–—]|-(?=\s+dla\s+)', item, maxsplit=1)
                test_data['test_name'] = parts[0].strip()
                
                if len(parts) > 1:
                    conditions_text = parts[1].strip()
                    test_data['conditions'].append(conditions_text)
                    
                    # Wyciągnij nazwy leków
                    drugs = extract_drug_names(conditions_text)
                    test_data['specific_drugs'].extend(drugs)
            else:
                test_data['test_name'] = item.strip()
            
            # Wyciągnij częstotliwość jeśli jest
            frequency = extract_frequency(item)
            if frequency:
                test_data['frequency'] = frequency
            
            parsed.append(test_data)
        
        return parsed

    def classify_test_type(text: str) -> str:
        """
        Klasyfikuje typ badania na podstawie słów kluczowych
        """
        
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['morfologia', 'krwi', 'oznaczenie stężenia', 'oznaczenie aktywności']):
            return 'laboratory_blood'
        elif any(word in text_lower for word in ['badanie ogólne moczu', 'moczu']):
            return 'laboratory_urine'
        elif any(word in text_lower for word in ['tk', 'tomografia', 'rezonans', 'mr', 'mri', 'rtg']):
            return 'imaging'
        elif 'ekg' in text_lower or 'elektrokardiogram' in text_lower:
            return 'ekg'
        elif 'ciśnienie' in text_lower or 'ciśnienia tętniczego' in text_lower:
            return 'blood_pressure'
        elif 'test ciążowy' in text_lower:
            return 'pregnancy_test'
        elif 'histologiczne' in text_lower:
            return 'histology'
        elif any(word in text_lower for word in ['msi-h', 'dmmr', 'mikrosatelitarnej']):
            return 'molecular'
        else:
            return 'other'

    def extract_drug_names(text: str) -> List[str]:
        """
        Wyciąga nazwy leków ze zdania warunkowego
        """
        
        drugs = []
        
        # Pattern: "dla <lek1>, <lek2> oraz <lek3>"
        drug_section = re.search(r'dla ([^.;]+)', text, re.IGNORECASE)
        
        if drug_section:
            drug_text = drug_section.group(1)
            
            # Podziel po przecinkach i "oraz"/"i"
            drug_candidates = re.split(r',|\s+oraz\s+|\s+i\s+', drug_text)
            
            for candidate in drug_candidates:
                candidate = candidate.strip()
                
                # Wyczyść z dodatkowych słów
                candidate = re.sub(r'\s+w\s+skojarzeniu.*$', '', candidate)
                candidate = re.sub(r'\s+stosowanych.*$', '', candidate)
                
                if candidate and len(candidate) > 3:  # Minimum 4 znaki dla nazwy leku
                    drugs.append(candidate)
        
        return drugs

    def extract_frequency(text: str) -> Optional[Dict]:
        """
        Wyciąga informacje o częstotliwości wykonywania badań
        """
        
        frequency_patterns = [
            (r'co (\d+)(?:-(\d+))? tygodni', 'weeks'),
            (r'przed (?:rozpoczęciem )?każdego cyklu', 'every_cycle'),
            (r'przed (?:rozpoczęciem )?co drugiego cyklu', 'every_second_cycle'),
            (r'nie rzadziej niż co (\d+) tygodni', 'min_weeks'),
            (r'przed każdym cyklem', 'every_cycle'),
        ]
        
        for pattern, freq_type in frequency_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if freq_type in ['weeks', 'min_weeks']:
                    return {
                        'type': freq_type,
                        'value': int(match.group(1)),
                        'range': int(match.group(2)) if match.lastindex > 1 and match.group(2) else None
                    }
                else:
                    return {'type': freq_type}
        
        return None

    def create_monitoring_structure(self, sections: List[Dict]) -> Dict:
        """
        Tworzy hierarchiczną strukturę danych z sekcji
        """
        
        structure = {
            'qualification_tests': None,
            'safety_monitoring': None,
            'efficacy_monitoring': None,
            'program_monitoring': None
        }
        
        for section in sections:
            title_lower = section['title'].lower()
            
            if 'kwalifikacji' in title_lower or 'kwalifikacja' in title_lower:
                structure['qualification_tests'] = {
                    'title': section['title'],
                    'tests': section['parsed_items'],
                    'summary': summarize_tests(section['parsed_items'])
                }
            
            elif 'bezpieczeństwa' in title_lower or 'bezpieczeństwo' in title_lower:
                structure['safety_monitoring'] = {
                    'title': section['title'],
                    'tests': section['parsed_items'],
                    'drug_specific': extract_drug_specific_monitoring(section['items']),
                    'summary': summarize_tests(section['parsed_items'])
                }
            
            elif 'skuteczności' in title_lower or 'skuteczność' in title_lower:
                structure['efficacy_monitoring'] = {
                    'title': section['title'],
                    'tests': section['parsed_items'],
                    'response_criteria': extract_response_criteria(section['items']),
                    'summary': summarize_tests(section['parsed_items'])
                }
            
            elif 'programu' in title_lower or 'monitorowanie programu' in title_lower:
                structure['program_monitoring'] = {
                    'title': section['title'],
                    'requirements': section['items']
                }
        
        return structure

    def extract_drug_specific_monitoring(self, items: List[str]) -> Dict[str, List[str]]:
        """
        Wyciąga wymagania specyficzne dla konkretnych leków
        """
        
        drug_specific = {}
        
        for item in items:
            # Szukaj zdań rozpoczynających się od "W przypadku leczenia:"
            if 'w przypadku leczenia' in item.lower():
                # Wyciągnij nazwę leku i częstotliwość
                drug_match = re.search(
                    r'([a-ząćęłńóśźż]+(?:\s+w\s+skojarzeniu\s+z\s+[a-ząćęłńóśźż]+)?)\s+powyższe\s+badania',
                    item,
                    re.IGNORECASE
                )
                
                if drug_match:
                    drug_name = drug_match.group(1).strip()
                    
                    # Wyciągnij częstotliwość
                    freq_match = re.search(
                        r'wykonuje się\s+(.+?)(?:\.|$)',
                        item,
                        re.IGNORECASE
                    )
                    
                    if freq_match:
                        frequency = freq_match.group(1).strip()
                        
                        if drug_name not in drug_specific:
                            drug_specific[drug_name] = []
                        
                        drug_specific[drug_name].append(frequency)
        
        return drug_specific

    def extract_response_criteria( items: List[str]) -> List[str]:
        """
        Wyciąga kryteria odpowiedzi na leczenie (CR, PR, SD, PD, OS, PFS)
        """
        
        criteria = []
        
        criteria_patterns = [
            r'całkowita.*?\(CR\)',
            r'częściowa odpowiedź.*?\(PR\)',
            r'stabilizacja.*?\(SD\)',
            r'progresja.*?\(PD\)',
            r'całkowite przeżycie.*?\(OS\)',
            r'czas do progresji.*?\(PFS\)',
        ]
        
        text = ' '.join(items)
        
        for pattern in criteria_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            criteria.extend(matches)
        
        return criteria

    def summarize_tests( parsed_items: List[Dict]) -> Dict:
        """
        Tworzy podsumowanie testów według typu
        """
        
        summary = {
            'total': len(parsed_items),
            'by_type': {},
            'drug_specific_count': 0,
            'general_count': 0
        }
        
        for item in parsed_items:
            test_type = item['type']
            
            if test_type not in summary['by_type']:
                summary['by_type'][test_type] = 0
            summary['by_type'][test_type] += 1
            
            if item['specific_drugs']:
                summary['drug_specific_count'] += 1
            else:
                summary['general_count'] += 1
        
        return summary

    def prepare_monitoring_for_streamlit(parsed_data: Dict) -> tuple:
        """
        Przygotowuje dane do wyświetlenia w Streamlit
        Zwraca: (df_tests, df_frequency, summary_dict)
        """
        
        # DataFrame 1: Wszystkie testy
        test_rows = []
        
        for section_key, section_data in parsed_data['full_structure'].items():
            if section_data and 'tests' in section_data:
                for test in section_data['tests']:
                    row = {
                        'Sekcja': section_data['title'],
                        'Badanie': test['test_name'],
                        'Typ': translate_test_type(test['type']),
                        'Leki': ', '.join(test['specific_drugs']) if test['specific_drugs'] else 'Wszystkie',
                        'Częstotliwość': format_frequency(test['frequency']) if test['frequency'] else 'Zgodnie z programem'
                    }
                    test_rows.append(row)
        
        df_tests = pd.DataFrame(test_rows)
        
        # DataFrame 2: Częstotliwość monitorowania dla leków
        freq_rows = []
        
        if parsed_data['full_structure'].get('safety_monitoring'):
            drug_specific = parsed_data['full_structure']['safety_monitoring'].get('drug_specific', {})
            
            for drug, frequencies in drug_specific.items():
                row = {
                    'Lek': drug,
                    'Częstotliwość monitorowania': ', '.join(frequencies)
                }
                freq_rows.append(row)
        
        df_frequency = pd.DataFrame(freq_rows) if freq_rows else None
        
        # Podsumowanie
        summary = {}
        for section_key, section_data in parsed_data['full_structure'].items():
            if section_data and 'summary' in section_data:
                summary[section_data['title']] = section_data['summary']
        
        return df_tests, df_frequency, summary

    def translate_test_type(test_type: str) -> str:
        """
        Tłumaczy typ testu na czytelną nazwę
        """
        
        translations = {
            'laboratory_blood': 'Badanie krwi',
            'laboratory_urine': 'Badanie moczu',
            'imaging': 'Badanie obrazowe',
            'ekg': 'EKG',
            'blood_pressure': 'Pomiar ciśnienia',
            'pregnancy_test': 'Test ciążowy',
            'histology': 'Badanie histologiczne',
            'molecular': 'Badanie molekularne',
            'other': 'Inne'
        }
        
        return translations.get(test_type, test_type)

    def format_frequency( frequency: Optional[Dict]) -> str:
        """
        Formatuje częstotliwość do czytelnej formy
        """
        
        if not frequency:
            return 'Nie określono'
        
        freq_type = frequency['type']
        
        if freq_type == 'weeks':
            value = frequency['value']
            range_val = frequency.get('range')
            if range_val:
                return f"Co {value}-{range_val} tygodni"
            return f"Co {value} tygodni"
        
        elif freq_type == 'min_weeks':
            return f"Nie rzadziej niż co {frequency['value']} tygodni"
        
        elif freq_type == 'every_cycle':
            return "Przed każdym cyklem"
        
        elif freq_type == 'every_second_cycle':
            return "Przed co drugim cyklem"
        
        return str(frequency)
    
    def pragmatic_parse_program(self, text: str) -> Dict:
        """
        Parsuje kolumnę 1: opis programu, listę substancji, kryteria ogólne i szczegółowe,
        kryteria wyłączenia oraz czas leczenia.
        """
        result = {
            'program_description': None,
            'substances': {},
            'general_criteria': [],
            'specific_criteria': {},
            'treatment_duration': None,
            'exclusion_criteria': []
        }

        lines = [l.strip() for l in text.strip().split('\n') if l.strip()]

        # 1. Opis programu (pierwsza linia do "substancjami:")
        desc_lines = []
        for line in lines:
            desc_lines.append(line)
            if 'substancjami:' in line:
                break
        result['program_description'] = ' '.join(desc_lines)

        # 2. Lista substancji w zakresach I i II
        substances = {'chemotherapy_targeted': [], 'immunotherapy': []}
        section = None
        for line in lines:
            if line.startswith('I.'):
                section = 'chemotherapy_targeted'
                continue
            if line.startswith('II.'):
                section = 'immunotherapy'
                continue
            if section and line.endswith(';'):
                substances[section].append(line.rstrip(';'))
            if section and not line.endswith(';') and section in result['substances'] and substances[section]:
                # Koniec listy substancji
                section = None
        result['substances'] = substances

        # 3. Kryteria ogólne (od "Ogólne kryteria kwalifikacji" do pustej linii lub "Szczegółowe")
        general = []
        specific = {}
        mode = None
        current_substance = None

        for line in lines:
            if line.lower().startswith('ogólne kryteria kwalifikacji'):
                mode = 'general'
                continue
            if line.lower().startswith('szczegółowe kryteria kwalifikacji'):
                mode = 'specific'
                continue
            if mode == 'general':
                if line.endswith(';'):
                    general.append(line.rstrip(';'))
            elif mode == 'specific':
                # Rozdziel substancje od ich kryteriów
                match = re.match(r'([a-ząćęłńóśźż ]+)(?:, niwolumabem.*| w skojarzeniu.*)?', line, re.IGNORECASE)
                if match:
                    # Nowa substancja
                    current_substance = match.group(1).strip()
                    specific[current_substance] = []
                    # Reszta linii jako kryterium
                    rest = line[len(match.group(0)):].strip(' :;')
                    if rest:
                        specific[current_substance].append(rest)
                else:
                    if current_substance and line.endswith(';'):
                        specific[current_substance].append(line.rstrip(';'))

        result['general_criteria'] = general
        result['specific_criteria'] = specific

        # 4. Czas leczenia (linia zaczynająca się od "Określenie czasu leczenia")
        dur_match = re.search(r'Określenie czasu leczenia\s*(.*)', text, re.IGNORECASE)
        if dur_match:
            result['treatment_duration'] = dur_match.group(1).strip()

        # 5. Kryteria wyłączenia (od "Kryteria wyłączenia" do końca)
        excl_mode = False
        exclusions = []
        for line in lines:
            if line.lower().startswith('kryteria wyłączenia'):
                excl_mode = True
                continue
            if excl_mode and line.endswith(';'):
                exclusions.append(line.rstrip(';'))
        result['exclusion_criteria'] = exclusions

        return result

    def prepare_program_for_streamlit(parsed: Dict) -> pd.DataFrame:
        """
        Przygotowuje DataFrame przedstawiający substancje i ich przypisane kryteria.
        """
        rows = []
        for part, drugs in parsed['substances'].items():
            for drug in drugs:
                row = {
                    'Zakres': 'Chemioterapia/Celowane' if part == 'chemotherapy_targeted' else 'Immunoterapia',
                    'Substancja': drug
                }
                rows.append(row)
        df_substances = pd.DataFrame(rows)

        # Kryteria ogólne
        df_general = pd.DataFrame({
            'Kryterium ogólne': parsed['general_criteria']
        })

        # Kryteria szczegółowe
        spec_rows = []
        for drug, crits in parsed['specific_criteria'].items():
            for c in crits:
                spec_rows.append({'Substancja': drug, 'Kryterium szczegółowe': c})
        df_specific = pd.DataFrame(spec_rows)

        return df_substances, df_general, df_specific