#!/usr/bin/env python3
"""
Synthetic CMS 1500 Claims Data Generator with Upcoding Patterns

This script generates synthetic healthcare claims data in CMS 1500 format
with realistic upcoding patterns for fraud detection training.

Features:
- E&M (Evaluation & Management) upcoding patterns
- Surgical procedure upcoding patterns
- Binary flags for upcoding detection
- Provider-specific upcoding tendencies

Author: AI Assistant
Date: 2025
"""

import random
import csv
import json
from datetime import datetime, timedelta
from faker import Faker
import uuid

# Initialize Faker for generating realistic data
fake = Faker()

class CMS1500UpcodingGenerator:
    def __init__(self):
        self.setup_code_tables()
        self.setup_provider_data()
        self.setup_upcoding_rules()
        
    def setup_code_tables(self):
        """Initialize medical code tables with common codes and complexity levels"""
        
        # E&M CPT Codes with complexity hierarchy (for upcoding detection)
        self.em_codes = {
            # New Patient Office Visits (ascending complexity/reimbursement)
            '99201': {'description': 'Office visit, new patient, problem focused', 'price_range': (75, 125), 'complexity': 1, 'discontinued': True},
            '99202': {'description': 'Office visit, new patient, expanded problem focused', 'price_range': (100, 175), 'complexity': 2},
            '99203': {'description': 'Office visit, new patient, detailed', 'price_range': (175, 275), 'complexity': 3},
            '99204': {'description': 'Office visit, new patient, comprehensive moderate', 'price_range': (275, 400), 'complexity': 4},
            '99205': {'description': 'Office visit, new patient, comprehensive high', 'price_range': (350, 500), 'complexity': 5},
            
            # Established Patient Office Visits
            '99211': {'description': 'Office visit, established patient, minimal', 'price_range': (50, 75), 'complexity': 1},
            '99212': {'description': 'Office visit, established patient, problem focused', 'price_range': (75, 125), 'complexity': 2},
            '99213': {'description': 'Office visit, established patient, expanded', 'price_range': (125, 200), 'complexity': 3},
            '99214': {'description': 'Office visit, established patient, detailed', 'price_range': (175, 275), 'complexity': 4},
            '99215': {'description': 'Office visit, established patient, comprehensive', 'price_range': (250, 375), 'complexity': 5},
            
            # Emergency Department Visits
            '99281': {'description': 'ED visit, problem focused', 'price_range': (150, 250), 'complexity': 1},
            '99282': {'description': 'ED visit, expanded problem focused', 'price_range': (200, 300), 'complexity': 2},
            '99283': {'description': 'ED visit, detailed', 'price_range': (300, 450), 'complexity': 3},
            '99284': {'description': 'ED visit, comprehensive moderate', 'price_range': (450, 650), 'complexity': 4},
            '99285': {'description': 'ED visit, comprehensive high', 'price_range': (600, 900), 'complexity': 5},
            
            # Hospital Visits
            '99221': {'description': 'Initial hospital care, detailed', 'price_range': (200, 350), 'complexity': 3},
            '99222': {'description': 'Initial hospital care, comprehensive moderate', 'price_range': (300, 450), 'complexity': 4},
            '99223': {'description': 'Initial hospital care, comprehensive high', 'price_range': (400, 600), 'complexity': 5},
        }
        
        # Surgical CPT Codes with complexity hierarchy
        self.surgical_codes = {
            # Arthroscopic procedures (knee) - ascending complexity
            '29870': {'description': 'Arthroscopy, knee, diagnostic', 'price_range': (1500, 2500), 'complexity': 1},
            '29871': {'description': 'Arthroscopy, knee, surgical; for infection', 'price_range': (2000, 3000), 'complexity': 2},
            '29873': {'description': 'Arthroscopy, knee, surgical; lateral release', 'price_range': (2500, 3500), 'complexity': 2},
            '29874': {'description': 'Arthroscopy, knee, surgical; removal of loose body', 'price_range': (2500, 3500), 'complexity': 3},
            '29875': {'description': 'Arthroscopy, knee, surgical; synovectomy', 'price_range': (3000, 4000), 'complexity': 3},
            '29876': {'description': 'Arthroscopy, knee, surgical; synovectomy major', 'price_range': (3500, 4500), 'complexity': 4},
            '29877': {'description': 'Arthroscopy, knee, surgical; debridement/shaving', 'price_range': (2800, 3800), 'complexity': 3},
            '29879': {'description': 'Arthroscopy, knee, surgical; abrasion arthroplasty', 'price_range': (3200, 4200), 'complexity': 4},
            '29880': {'description': 'Arthroscopy, knee, surgical; meniscectomy', 'price_range': (3500, 4500), 'complexity': 4},
            '29881': {'description': 'Arthroscopy, knee, surgical; meniscectomy major', 'price_range': (4000, 5500), 'complexity': 5},
            '29882': {'description': 'Arthroscopy, knee, surgical; meniscus repair', 'price_range': (4500, 6000), 'complexity': 5},
            '29883': {'description': 'Arthroscopy, knee, surgical; meniscus transplant', 'price_range': (8000, 12000), 'complexity': 6},
            
            # General Surgery procedures
            '10060': {'description': 'Incision and drainage of abscess, simple', 'price_range': (150, 300), 'complexity': 1},
            '10061': {'description': 'Incision and drainage of abscess, complicated', 'price_range': (250, 450), 'complexity': 2},
            '11401': {'description': 'Excision, benign lesion, trunk/arms/legs 0.6-1.0 cm', 'price_range': (200, 350), 'complexity': 2},
            '11402': {'description': 'Excision, benign lesion, trunk/arms/legs 1.1-2.0 cm', 'price_range': (300, 500), 'complexity': 3},
            '11403': {'description': 'Excision, benign lesion, trunk/arms/legs 2.1-3.0 cm', 'price_range': (400, 650), 'complexity': 4},
            '11404': {'description': 'Excision, benign lesion, trunk/arms/legs 3.1-4.0 cm', 'price_range': (500, 800), 'complexity': 5},
            
            # Cardiovascular procedures
            '33533': {'description': 'Coronary artery bypass, single graft', 'price_range': (25000, 35000), 'complexity': 4},
            '33534': {'description': 'Coronary artery bypass, single graft, arterial', 'price_range': (30000, 42000), 'complexity': 5},
            '33535': {'description': 'Coronary artery bypass, two coronary arteries', 'price_range': (35000, 50000), 'complexity': 5},
            '33536': {'description': 'Coronary artery bypass, three coronary arteries', 'price_range': (40000, 55000), 'complexity': 6},
        }
        
        # Non-surgical procedures for control group
        self.diagnostic_codes = {
            '85025': {'description': 'Complete blood count with differential', 'price_range': (25, 45), 'complexity': 1},
            '80053': {'description': 'Comprehensive metabolic panel', 'price_range': (30, 60), 'complexity': 1},
            '93000': {'description': 'Electrocardiogram', 'price_range': (50, 100), 'complexity': 1},
            '71020': {'description': 'Chest X-ray, 2 views', 'price_range': (80, 150), 'complexity': 2},
            '73721': {'description': 'MRI lower extremity without contrast', 'price_range': (1200, 2000), 'complexity': 3},
            '45378': {'description': 'Colonoscopy, diagnostic', 'price_range': (800, 1500), 'complexity': 3},
            '36415': {'description': 'Venipuncture', 'price_range': (15, 30), 'complexity': 1},
            '90471': {'description': 'Immunization administration', 'price_range': (25, 40), 'complexity': 1},
            '81001': {'description': 'Urinalysis', 'price_range': (20, 35), 'complexity': 1}
        }
        
        # Common ICD-10 Diagnosis Codes with severity indicators
        self.icd10_codes = {
            # Simple/routine conditions
            'Z00.00': {'description': 'Encounter for general adult medical examination without abnormal findings', 'severity': 'low'},
            'J06.9': {'description': 'Acute upper respiratory infection, unspecified', 'severity': 'low'},
            'R51': {'description': 'Headache', 'severity': 'low'},
            'R05': {'description': 'Cough', 'severity': 'low'},
            'K59.00': {'description': 'Constipation, unspecified', 'severity': 'low'},
            'Z23': {'description': 'Encounter for immunization', 'severity': 'low'},
            'H52.4': {'description': 'Presbyopia', 'severity': 'low'},
            'L70.0': {'description': 'Acne vulgaris', 'severity': 'low'},
            
            # Moderate complexity conditions
            'I10': {'description': 'Essential hypertension', 'severity': 'moderate'},
            'E11.9': {'description': 'Type 2 diabetes mellitus without complications', 'severity': 'moderate'},
            'K21.9': {'description': 'Gastro-esophageal reflux disease without esophagitis', 'severity': 'moderate'},
            'M25.561': {'description': 'Pain in right knee', 'severity': 'moderate'},
            'M54.5': {'description': 'Low back pain', 'severity': 'moderate'},
            'N39.0': {'description': 'Urinary tract infection, site not specified', 'severity': 'moderate'},
            'E78.5': {'description': 'Hyperlipidemia, unspecified', 'severity': 'moderate'},
            
            # High complexity/severity conditions
            'R06.02': {'description': 'Shortness of breath', 'severity': 'high'},
            'R50.9': {'description': 'Fever, unspecified', 'severity': 'high'},
            'F32.9': {'description': 'Major depressive disorder, single episode, unspecified', 'severity': 'high'},
            'M79.89': {'description': 'Other specified soft tissue disorders', 'severity': 'high'},
            'S61.001A': {'description': 'Unspecified open wound of right thumb without damage to nail, initial encounter', 'severity': 'high'},
            'I21.9': {'description': 'Acute myocardial infarction, unspecified', 'severity': 'high'},
            'J44.1': {'description': 'Chronic obstructive pulmonary disease with acute exacerbation', 'severity': 'high'},
            'N18.6': {'description': 'End stage renal disease', 'severity': 'high'},
        }
        
        # Place of Service Codes
        self.pos_codes = {
            '11': 'Office',
            '12': 'Home',
            '21': 'Inpatient Hospital',
            '22': 'Outpatient Hospital',
            '23': 'Emergency Room - Hospital',
            '81': 'Independent Laboratory',
            '99': 'Other Place of Service'
        }
        
    def setup_provider_data(self):
        """Initialize provider data with upcoding tendencies"""
        self.providers = [
            # High-volume, clean providers (low upcoding rate)
            {'name': 'Dr. Sarah Johnson', 'npi': '1234567890', 'specialty': 'Family Practice', 'tax_id': '12-3456789', 'upcoding_tendency': 0.05},
            {'name': 'Dr. Michael Chen', 'npi': '2345678901', 'specialty': 'Internal Medicine', 'tax_id': '23-4567890', 'upcoding_tendency': 0.08},
            {'name': 'Dr. Lisa Thompson', 'npi': '5678901234', 'specialty': 'Dermatology', 'tax_id': '56-7890123', 'upcoding_tendency': 0.12},
            
            # Moderate upcoding providers
            {'name': 'Dr. Robert Davis', 'npi': '6789012345', 'specialty': 'Emergency Medicine', 'tax_id': '67-8901234', 'upcoding_tendency': 0.25},
            {'name': 'Dr. Jennifer Wilson', 'npi': '7890123456', 'specialty': 'Family Practice', 'tax_id': '78-9012345', 'upcoding_tendency': 0.30},
            {'name': 'Dr. Mark Anderson', 'npi': '8901234567', 'specialty': 'Internal Medicine', 'tax_id': '89-0123456', 'upcoding_tendency': 0.22},
            
            # High upcoding providers (potential fraud cases)
            {'name': 'Dr. David Wilson', 'npi': '4567890123', 'specialty': 'Orthopedics', 'tax_id': '45-6789012', 'upcoding_tendency': 0.45},
            {'name': 'Dr. Emily Rodriguez', 'npi': '3456789012', 'specialty': 'Cardiology', 'tax_id': '34-5678901', 'upcoding_tendency': 0.50},
            {'name': 'Dr. James Miller', 'npi': '9012345678', 'specialty': 'General Surgery', 'tax_id': '90-1234567', 'upcoding_tendency': 0.55},
            {'name': 'Dr. Karen Brown', 'npi': '0123456789', 'specialty': 'Emergency Medicine', 'tax_id': '01-2345678', 'upcoding_tendency': 0.60},
        ]
        
        self.facilities = [
            {'name': 'City Medical Center', 'address': '123 Main St', 'city': 'Springfield', 'state': 'IL', 'zip': '62701'},
            {'name': 'Regional Health Clinic', 'address': '456 Oak Ave', 'city': 'Chicago', 'state': 'IL', 'zip': '60601'},
            {'name': 'Suburban Family Practice', 'address': '789 Elm Dr', 'city': 'Naperville', 'state': 'IL', 'zip': '60540'},
            {'name': 'Metro Specialty Center', 'address': '321 Pine St', 'city': 'Peoria', 'state': 'IL', 'zip': '61602'},
            {'name': 'Community Health Partners', 'address': '654 Cedar Ln', 'city': 'Rockford', 'state': 'IL', 'zip': '61101'}
        ]
        
        self.insurance_plans = [
            {'name': 'Blue Cross Blue Shield', 'payer_id': 'BCBS001', 'group_number': 'GRP12345'},
            {'name': 'Aetna', 'payer_id': 'AETNA01', 'group_number': 'GRP23456'},
            {'name': 'UnitedHealthcare', 'payer_id': 'UHC0001', 'group_number': 'GRP34567'},
            {'name': 'Cigna', 'payer_id': 'CIGNA01', 'group_number': 'GRP45678'},
            {'name': 'Humana', 'payer_id': 'HUMANA1', 'group_number': 'GRP56789'},
            {'name': 'Medicare', 'payer_id': 'MEDICARE', 'group_number': 'MEDICARE'},
            {'name': 'Medicaid', 'payer_id': 'MEDICAID', 'group_number': 'MEDICAID'}
        ]
    
    def setup_upcoding_rules(self):
        """Setup rules for realistic upcoding patterns"""
        
        # E&M upcoding rules: inappropriate complexity upgrades
        self.em_upcoding_rules = {
            # New patient visits - upcode from appropriate level
            '99202': ['99203', '99204'],  # Upcode to higher complexity
            '99203': ['99204', '99205'],
            '99204': ['99205'],
            
            # Established patient visits
            '99211': ['99212', '99213'],
            '99212': ['99213', '99214'],
            '99213': ['99214', '99215'],
            '99214': ['99215'],
            
            # Emergency department visits
            '99281': ['99282', '99283'],
            '99282': ['99283', '99284'],
            '99283': ['99284', '99285'],
            '99284': ['99285'],
        }
        
        # Surgical upcoding rules: upgrade to more complex procedures
        self.surgical_upcoding_rules = {
            # Arthroscopic procedures
            '29870': ['29871', '29873', '29874'],  # Diagnostic to surgical
            '29871': ['29874', '29875'],
            '29873': ['29874', '29877'],
            '29874': ['29875', '29877', '29880'],
            '29877': ['29880', '29881'],
            '29880': ['29881', '29882'],
            
            # Lesion excisions
            '11401': ['11402', '11403'],
            '11402': ['11403', '11404'],
            '11403': ['11404'],
            
            # Abscess drainage
            '10060': ['10061'],
        }
    
    def determine_appropriate_em_code(self, diagnosis_codes, patient_age):
        """Determine appropriate E&M code based on diagnosis complexity"""
        severity_scores = [self.icd10_codes.get(dx, {}).get('severity', 'low') for dx in diagnosis_codes]
        
        # Calculate complexity score
        complexity_score = 0
        for severity in severity_scores:
            if severity == 'low':
                complexity_score += 1
            elif severity == 'moderate':
                complexity_score += 2
            elif severity == 'high':
                complexity_score += 3
        
        # Adjust for patient age (elderly patients typically have higher complexity)
        if patient_age > 65:
            complexity_score += 1
        
        # Map complexity to appropriate E&M code
        if complexity_score <= 2:
            return random.choice(['99211', '99212', '99202'])  # Low complexity
        elif complexity_score <= 4:
            return random.choice(['99213', '99203'])  # Moderate complexity
        elif complexity_score <= 6:
            return random.choice(['99214', '99204'])  # High complexity
        else:
            return random.choice(['99215', '99205'])  # Very high complexity
    
    def apply_upcoding(self, original_code, provider_tendency, procedure_type):
        """Apply upcoding based on provider tendency and rules"""
        
        # Determine if this claim should be upcoded
        if random.random() > provider_tendency:
            return original_code, False  # No upcoding
        
        # Apply appropriate upcoding rules
        if procedure_type == 'em' and original_code in self.em_upcoding_rules:
            upcoded_options = self.em_upcoding_rules[original_code]
            upcoded_code = random.choice(upcoded_options)
            return upcoded_code, True
            
        elif procedure_type == 'surgical' and original_code in self.surgical_upcoding_rules:
            upcoded_options = self.surgical_upcoding_rules[original_code]
            upcoded_code = random.choice(upcoded_options)
            return upcoded_code, True
        
        return original_code, False
    
    def generate_patient_data(self):
        """Generate synthetic patient information"""
        gender = random.choice(['M', 'F'])
        birth_date = fake.date_of_birth(minimum_age=18, maximum_age=85)
        
        patient = {
            'patient_id': fake.unique.random_number(digits=8),
            'first_name': fake.first_name_male() if gender == 'M' else fake.first_name_female(),
            'last_name': fake.last_name(),
            'middle_initial': fake.random_letter().upper(),
            'date_of_birth': birth_date.strftime('%m/%d/%Y'),
            'gender': gender,
            'age': (datetime.now().date() - birth_date).days // 365,
            'address': fake.street_address(),
            'city': fake.city(),
            'state': fake.state_abbr(),
            'zip_code': fake.zipcode(),
            'phone': fake.phone_number(),
            'ssn': fake.ssn(),
            'member_id': fake.random_number(digits=10, fix_len=True)
        }
        return patient
    
    def generate_claim_data(self):
        """Generate a complete CMS 1500 claim with potential upcoding"""
        patient = self.generate_patient_data()
        provider = random.choice(self.providers)
        facility = random.choice(self.facilities)
        insurance = random.choice(self.insurance_plans)
        
        # Generate service date (within last 90 days)
        service_date = fake.date_between(start_date='-90d', end_date='today')
        
        # Select diagnosis codes (1-4 diagnoses)
        num_diagnoses = random.randint(1, 4)
        diagnosis_codes = random.sample(list(self.icd10_codes.keys()), num_diagnoses)
        
        # Determine primary procedure type and generate appropriate codes
        procedure_type = random.choices(['em', 'surgical', 'diagnostic'], weights=[0.6, 0.25, 0.15])[0]
        
        upcoding_flags = []
        em_upcoding_flag = False
        surgical_upcoding_flag = False
        
        procedure_lines = []
        total_charges = 0
        
        if procedure_type == 'em':
            # Generate E&M visit
            appropriate_em_code = self.determine_appropriate_em_code(diagnosis_codes, patient['age'])
            final_em_code, was_upcoded = self.apply_upcoding(appropriate_em_code, provider['upcoding_tendency'], 'em')
            
            if was_upcoded:
                em_upcoding_flag = True
                upcoding_flags.append(f"E&M upcoded from {appropriate_em_code} to {final_em_code}")
            
            # Get code details
            code_info = self.em_codes[final_em_code]
            min_price, max_price = code_info['price_range']
            charges = round(random.uniform(min_price, max_price), 2)
            total_charges += charges
            
            procedure_line = {
                'line_number': 1,
                'service_date': service_date.strftime('%m/%d/%Y'),
                'place_of_service': '11',  # Office
                'procedure_code': final_em_code,
                'procedure_description': code_info['description'],
                'modifier': '',
                'diagnosis_pointer': ','.join([str(j+1) for j in range(min(len(diagnosis_codes), 2))]),
                'charges': charges,
                'units': 1,
                'rendering_provider_npi': provider['npi'],
                'appropriate_code': appropriate_em_code,
                'complexity_level': code_info['complexity']
            }
            procedure_lines.append(procedure_line)
            
            # Add some ancillary services (labs, etc.)
            if random.random() < 0.3:
                ancillary_codes = random.sample(list(self.diagnostic_codes.keys()), random.randint(1, 2))
                for i, anc_code in enumerate(ancillary_codes):
                    code_info = self.diagnostic_codes[anc_code]
                    min_price, max_price = code_info['price_range']
                    charges = round(random.uniform(min_price, max_price), 2)
                    total_charges += charges
                    
                    procedure_line = {
                        'line_number': i + 2,
                        'service_date': service_date.strftime('%m/%d/%Y'),
                        'place_of_service': '11',
                        'procedure_code': anc_code,
                        'procedure_description': code_info['description'],
                        'modifier': '',
                        'diagnosis_pointer': '1',
                        'charges': charges,
                        'units': 1,
                        'rendering_provider_npi': provider['npi'],
                        'appropriate_code': anc_code,
                        'complexity_level': code_info['complexity']
                    }
                    procedure_lines.append(procedure_line)
        
        elif procedure_type == 'surgical':
            # Generate surgical procedure
            surgical_codes_list = list(self.surgical_codes.keys())
            # Weight selection towards less complex procedures (more likely to be appropriate)
            weights = [6-self.surgical_codes[code]['complexity'] for code in surgical_codes_list]
            appropriate_surgical_code = random.choices(surgical_codes_list, weights=weights)[0]
            
            final_surgical_code, was_upcoded = self.apply_upcoding(appropriate_surgical_code, provider['upcoding_tendency'], 'surgical')
            
            if was_upcoded:
                surgical_upcoding_flag = True
                upcoding_flags.append(f"Surgical upcoded from {appropriate_surgical_code} to {final_surgical_code}")
            
            # Get code details
            code_info = self.surgical_codes[final_surgical_code]
            min_price, max_price = code_info['price_range']
            charges = round(random.uniform(min_price, max_price), 2)
            total_charges += charges
            
            procedure_line = {
                'line_number': 1,
                'service_date': service_date.strftime('%m/%d/%Y'),
                'place_of_service': random.choice(['21', '22', '23']),  # Hospital settings
                'procedure_code': final_surgical_code,
                'procedure_description': code_info['description'],
                'modifier': random.choice(['', 'RT', 'LT', '59']) if random.random() < 0.4 else '',
                'diagnosis_pointer': ','.join([str(j+1) for j in range(min(len(diagnosis_codes), 2))]),
                'charges': charges,
                'units': 1,
                'rendering_provider_npi': provider['npi'],
                'appropriate_code': appropriate_surgical_code,
                'complexity_level': code_info['complexity']
            }
            procedure_lines.append(procedure_line)
        
        else:  # diagnostic
            # Generate diagnostic procedures (no upcoding typically)
            num_procedures = random.randint(1, 3)
            selected_codes = random.sample(list(self.diagnostic_codes.keys()), num_procedures)
            
            for i, diag_code in enumerate(selected_codes):
                code_info = self.diagnostic_codes[diag_code]
                min_price, max_price = code_info['price_range']
                charges = round(random.uniform(min_price, max_price), 2)
                total_charges += charges
                
                procedure_line = {
                    'line_number': i + 1,
                    'service_date': service_date.strftime('%m/%d/%Y'),
                    'place_of_service': random.choice(['11', '22', '81']),
                    'procedure_code': diag_code,
                    'procedure_description': code_info['description'],
                    'modifier': '',
                    'diagnosis_pointer': '1',
                    'charges': charges,
                    'units': 1,
                    'rendering_provider_npi': provider['npi'],
                    'appropriate_code': diag_code,
                    'complexity_level': code_info['complexity']
                }
                procedure_lines.append(procedure_line)
        
        # Calculate fraud risk score based on multiple factors
        fraud_risk_score = 0
        if em_upcoding_flag:
            fraud_risk_score += 0.4
        if surgical_upcoding_flag:
            fraud_risk_score += 0.5
        
        # Add provider-specific risk
        fraud_risk_score += provider['upcoding_tendency'] * 0.3
        
        # Add complexity mismatch risk
        for line in procedure_lines:
            if 'appropriate_code' in line and line['procedure_code'] != line['appropriate_code']:
                # Calculate complexity difference
                if line['procedure_code'] in self.em_codes and line['appropriate_code'] in self.em_codes:
                    complexity_diff = self.em_codes[line['procedure_code']]['complexity'] - self.em_codes[line['appropriate_code']]['complexity']
                elif line['procedure_code'] in self.surgical_codes and line['appropriate_code'] in self.surgical_codes:
                    complexity_diff = self.surgical_codes[line['procedure_code']]['complexity'] - self.surgical_codes[line['appropriate_code']]['complexity']
                else:
                    complexity_diff = 0
                
                if complexity_diff > 0:
                    fraud_risk_score += complexity_diff * 0.15
        
        # Normalize fraud risk score to 0-1 range
        fraud_risk_score = min(1.0, fraud_risk_score)
        
        # Binary flags for fraud detection
        overall_upcoding_flag = em_upcoding_flag or surgical_upcoding_flag
        
        # Generate claim header information
        claim = {
            'claim_id': str(uuid.uuid4()),
            'claim_number': fake.random_number(digits=12, fix_len=True),
            'patient_control_number': fake.random_number(digits=8, fix_len=True),
            'type_of_service': procedure_type,
            'place_of_service': procedure_lines[0]['place_of_service'],
            'submission_date': datetime.now().strftime('%m/%d/%Y'),
            'onset_date': service_date.strftime('%m/%d/%Y') if random.random() < 0.3 else '',
            'total_charges': round(total_charges, 2),
            'amount_paid': round(total_charges * random.uniform(0.7, 0.95), 2),
            'patient_signature_on_file': 'Y',
            'assignment_of_benefits': random.choice(['Y', 'N']),
            'prior_authorization': fake.random_number(digits=10) if random.random() < 0.2 else '',
            
            # Upcoding and fraud detection fields
            'em_upcoding_flag': em_upcoding_flag,
            'surgical_upcoding_flag': surgical_upcoding_flag,
            'overall_upcoding_flag': overall_upcoding_flag,
            'fraud_risk_score': round(fraud_risk_score, 3),
            'provider_upcoding_tendency': provider['upcoding_tendency'],
            'upcoding_details': upcoding_flags,
            
            # Patient information
            'patient': patient,
            
            # Insurance information
            'primary_insurance': {
                'insurance_name': insurance['name'],
                'payer_id': insurance['payer_id'],
                'group_number': insurance['group_number'],
                'policy_number': fake.random_number(digits=12, fix_len=True)
            },
            
            # Provider information
            'billing_provider': provider,
            
            # Facility information
            'service_facility': facility,
            
            # Diagnosis information
            'diagnoses': [
                {
                    'code': code,
                    'description': self.icd10_codes[code]['description'],
                    'severity': self.icd10_codes[code]['severity'],
                    'pointer': i + 1
                }
                for i, code in enumerate(diagnosis_codes)
            ],
            
            # Procedure lines
            'procedure_lines': procedure_lines
        }
        
        return claim
    
    def generate_multiple_claims(self, num_claims=100):
        """Generate multiple claims with progress tracking"""
        claims = []
        print(f"Generating {num_claims:,} synthetic claims with upcoding patterns...")
        
        for i in range(num_claims):
            claim = self.generate_claim_data()
            claims.append(claim)
            
            if (i + 1) % 1000 == 0:
                print(f"Generated {i + 1:,} claims...")
        
        return claims
    
    def generate_claims_streaming(self, num_claims, batch_size=10000):
        """Generator function for memory-efficient large-scale claim generation"""
        print(f"Streaming generation of {num_claims:,} claims with upcoding patterns...")
        
        for i in range(0, num_claims, batch_size):
            current_batch_size = min(batch_size, num_claims - i)
            batch = []
            
            for j in range(current_batch_size):
                claim = self.generate_claim_data()
                batch.append(claim)
                
                if (i + j + 1) % 1000 == 0:
                    print(f"Generated {i + j + 1:,} claims...")
            
            yield batch
    
    def export_to_csv_with_fraud_flags(self, claims, filename='cms1500_claims_with_upcoding.csv'):
        """Export claims to CSV with upcoding and fraud detection columns"""
        fieldnames = [
            'claim_id', 'claim_number', 'patient_control_number', 'submission_date',
            'patient_id', 'patient_first_name', 'patient_last_name', 'patient_dob', 'patient_age',
            'patient_gender', 'patient_address', 'patient_city', 'patient_state', 'patient_zip',
            'insurance_name', 'policy_number', 'group_number',
            'provider_name', 'provider_npi', 'provider_specialty', 'provider_upcoding_tendency',
            'service_date', 'place_of_service', 'total_charges', 'amount_paid',
            'primary_diagnosis', 'primary_diagnosis_severity', 'secondary_diagnosis', 
            'procedure_codes', 'procedure_descriptions', 'procedure_type',
            # Fraud detection columns
            'em_upcoding_flag', 'surgical_upcoding_flag', 'overall_upcoding_flag',
            'fraud_risk_score', 'upcoding_details'
        ]
        
        print(f"Exporting {len(claims):,} claims with fraud flags to {filename}...")
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for claim in claims:
                row = {
                    'claim_id': claim['claim_id'],
                    'claim_number': claim['claim_number'],
                    'patient_control_number': claim['patient_control_number'],
                    'submission_date': claim['submission_date'],
                    'patient_id': claim['patient']['patient_id'],
                    'patient_first_name': claim['patient']['first_name'],
                    'patient_last_name': claim['patient']['last_name'],
                    'patient_dob': claim['patient']['date_of_birth'],
                    'patient_age': claim['patient']['age'],
                    'patient_gender': claim['patient']['gender'],
                    'patient_address': claim['patient']['address'],
                    'patient_city': claim['patient']['city'],
                    'patient_state': claim['patient']['state'],
                    'patient_zip': claim['patient']['zip_code'],
                    'insurance_name': claim['primary_insurance']['insurance_name'],
                    'policy_number': claim['primary_insurance']['policy_number'],
                    'group_number': claim['primary_insurance']['group_number'],
                    'provider_name': claim['billing_provider']['name'],
                    'provider_npi': claim['billing_provider']['npi'],
                    'provider_specialty': claim['billing_provider']['specialty'],
                    'provider_upcoding_tendency': claim['provider_upcoding_tendency'],
                    'service_date': claim['procedure_lines'][0]['service_date'],
                    'place_of_service': claim['place_of_service'],
                    'total_charges': claim['total_charges'],
                    'amount_paid': claim['amount_paid'],
                    'primary_diagnosis': claim['diagnoses'][0]['code'] if claim['diagnoses'] else '',
                    'primary_diagnosis_severity': claim['diagnoses'][0]['severity'] if claim['diagnoses'] else '',
                    'secondary_diagnosis': claim['diagnoses'][1]['code'] if len(claim['diagnoses']) > 1 else '',
                    'procedure_codes': ';'.join([line['procedure_code'] for line in claim['procedure_lines']]),
                    'procedure_descriptions': ';'.join([line['procedure_description'] for line in claim['procedure_lines']]),
                    'procedure_type': claim['type_of_service'],
                    # Fraud flags
                    'em_upcoding_flag': claim['em_upcoding_flag'],
                    'surgical_upcoding_flag': claim['surgical_upcoding_flag'],
                    'overall_upcoding_flag': claim['overall_upcoding_flag'],
                    'fraud_risk_score': claim['fraud_risk_score'],
                    'upcoding_details': ' | '.join(claim['upcoding_details']) if claim['upcoding_details'] else ''
                }
                writer.writerow(row)
        
        print(f"Export complete! Claims written to {filename}")
    
    def export_to_csv_streaming_with_fraud(self, num_claims, filename='cms1500_claims_with_upcoding.csv', batch_size=10000):
        """Export claims to CSV with streaming for large datasets, including fraud flags"""
        fieldnames = [
            'claim_id', 'claim_number', 'patient_control_number', 'submission_date',
            'patient_id', 'patient_first_name', 'patient_last_name', 'patient_dob', 'patient_age',
            'patient_gender', 'patient_address', 'patient_city', 'patient_state', 'patient_zip',
            'insurance_name', 'policy_number', 'group_number',
            'provider_name', 'provider_npi', 'provider_specialty', 'provider_upcoding_tendency',
            'service_date', 'place_of_service', 'total_charges', 'amount_paid',
            'primary_diagnosis', 'primary_diagnosis_severity', 'secondary_diagnosis', 
            'procedure_codes', 'procedure_descriptions', 'procedure_type',
            # Fraud detection columns
            'em_upcoding_flag', 'surgical_upcoding_flag', 'overall_upcoding_flag',
            'fraud_risk_score', 'upcoding_details'
        ]
        
        print(f"Exporting {num_claims:,} claims with fraud detection to {filename} using streaming...")
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            total_written = 0
            total_upcoded = 0
            em_upcoded = 0
            surgical_upcoded = 0
            
            for batch in self.generate_claims_streaming(num_claims, batch_size):
                for claim in batch:
                    row = {
                        'claim_id': claim['claim_id'],
                        'claim_number': claim['claim_number'],
                        'patient_control_number': claim['patient_control_number'],
                        'submission_date': claim['submission_date'],
                        'patient_id': claim['patient']['patient_id'],
                        'patient_first_name': claim['patient']['first_name'],
                        'patient_last_name': claim['patient']['last_name'],
                        'patient_dob': claim['patient']['date_of_birth'],
                        'patient_age': claim['patient']['age'],
                        'patient_gender': claim['patient']['gender'],
                        'patient_address': claim['patient']['address'],
                        'patient_city': claim['patient']['city'],
                        'patient_state': claim['patient']['state'],
                        'patient_zip': claim['patient']['zip_code'],
                        'insurance_name': claim['primary_insurance']['insurance_name'],
                        'policy_number': claim['primary_insurance']['policy_number'],
                        'group_number': claim['primary_insurance']['group_number'],
                        'provider_name': claim['billing_provider']['name'],
                        'provider_npi': claim['billing_provider']['npi'],
                        'provider_specialty': claim['billing_provider']['specialty'],
                        'provider_upcoding_tendency': claim['provider_upcoding_tendency'],
                        'service_date': claim['procedure_lines'][0]['service_date'],
                        'place_of_service': claim['place_of_service'],
                        'total_charges': claim['total_charges'],
                        'amount_paid': claim['amount_paid'],
                        'primary_diagnosis': claim['diagnoses'][0]['code'] if claim['diagnoses'] else '',
                        'primary_diagnosis_severity': claim['diagnoses'][0]['severity'] if claim['diagnoses'] else '',
                        'secondary_diagnosis': claim['diagnoses'][1]['code'] if len(claim['diagnoses']) > 1 else '',
                        'procedure_codes': ';'.join([line['procedure_code'] for line in claim['procedure_lines']]),
                        'procedure_descriptions': ';'.join([line['procedure_description'] for line in claim['procedure_lines']]),
                        'procedure_type': claim['type_of_service'],
                        # Fraud flags
                        'em_upcoding_flag': claim['em_upcoding_flag'],
                        'surgical_upcoding_flag': claim['surgical_upcoding_flag'],
                        'overall_upcoding_flag': claim['overall_upcoding_flag'],
                        'fraud_risk_score': claim['fraud_risk_score'],
                        'upcoding_details': ' | '.join(claim['upcoding_details']) if claim['upcoding_details'] else ''
                    }
                    writer.writerow(row)
                    total_written += 1
                    
                    # Track fraud statistics
                    if claim['overall_upcoding_flag']:
                        total_upcoded += 1
                    if claim['em_upcoding_flag']:
                        em_upcoded += 1
                    if claim['surgical_upcoding_flag']:
                        surgical_upcoded += 1
                
                # Flush and show progress
                csvfile.flush()
                print(f"Written {total_written:,} claims (Upcoded: {total_upcoded:,}, E&M: {em_upcoded:,}, Surgical: {surgical_upcoded:,})")
        
        upcoding_rate = (total_upcoded / total_written) * 100 if total_written > 0 else 0
        em_rate = (em_upcoded / total_written) * 100 if total_written > 0 else 0
        surgical_rate = (surgical_upcoded / total_written) * 100 if total_written > 0 else 0
        
        print(f"\nExport complete! {total_written:,} claims written to {filename}")
        print(f"Upcoding Statistics:")
        print(f"  Overall upcoding rate: {upcoding_rate:.1f}% ({total_upcoded:,} claims)")
        print(f"  E&M upcoding rate: {em_rate:.1f}% ({em_upcoded:,} claims)")
        print(f"  Surgical upcoding rate: {surgical_rate:.1f}% ({surgical_upcoded:,} claims)")
    
    def export_to_json(self, claims, filename='cms1500_claims_with_upcoding.json'):
        """Export claims to JSON format with fraud detection data"""
        with open(filename, 'w', encoding='utf-8') as jsonfile:
            json.dump(claims, jsonfile, indent=2, ensure_ascii=False, default=str)
        print(f"Claims with fraud detection data exported to {filename}")
    
    def analyze_fraud_patterns(self, claims):
        """Analyze and report fraud patterns in the generated claims"""
        total_claims = len(claims)
        em_upcoded = sum(1 for claim in claims if claim['em_upcoding_flag'])
        surgical_upcoded = sum(1 for claim in claims if claim['surgical_upcoding_flag'])
        overall_upcoded = sum(1 for claim in claims if claim['overall_upcoding_flag'])
        
        # Provider-specific analysis
        provider_stats = {}
        for claim in claims:
            provider_name = claim['billing_provider']['name']
            if provider_name not in provider_stats:
                provider_stats[provider_name] = {
                    'total_claims': 0,
                    'upcoded_claims': 0,
                    'em_upcoded': 0,
                    'surgical_upcoded': 0,
                    'total_charges': 0,
                    'upcoding_tendency': claim['provider_upcoding_tendency']
                }
            
            stats = provider_stats[provider_name]
            stats['total_claims'] += 1
            stats['total_charges'] += claim['total_charges']
            
            if claim['overall_upcoding_flag']:
                stats['upcoded_claims'] += 1
            if claim['em_upcoding_flag']:
                stats['em_upcoded'] += 1
            if claim['surgical_upcoding_flag']:
                stats['surgical_upcoded'] += 1
        
        # High-risk fraud scores
        high_risk_claims = [claim for claim in claims if claim['fraud_risk_score'] > 0.7]
        
        print(f"\n{'='*60}")
        print(f"FRAUD PATTERN ANALYSIS")
        print(f"{'='*60}")
        print(f"Total Claims Generated: {total_claims:,}")
        print(f"Overall Upcoding Rate: {(overall_upcoded/total_claims)*100:.1f}% ({overall_upcoded:,} claims)")
        print(f"E&M Upcoding Rate: {(em_upcoded/total_claims)*100:.1f}% ({em_upcoded:,} claims)")
        print(f"Surgical Upcoding Rate: {(surgical_upcoded/total_claims)*100:.1f}% ({surgical_upcoded:,} claims)")
        print(f"High Risk Claims (score > 0.7): {len(high_risk_claims):,}")
        
        print(f"\n{'PROVIDER ANALYSIS'}")
        print(f"{'Provider':<25} {'Claims':<8} {'Upcoded':<9} {'Rate':<7} {'Tendency':<9}")
        print(f"{'-'*65}")
        
        for provider, stats in sorted(provider_stats.items(), key=lambda x: x[1]['upcoded_claims']/x[1]['total_claims'], reverse=True):
            rate = (stats['upcoded_claims'] / stats['total_claims']) * 100
            print(f"{provider:<25} {stats['total_claims']:<8} {stats['upcoded_claims']:<9} {rate:<7.1f}% {stats['upcoding_tendency']:<9.3f}")
        
        return {
            'total_claims': total_claims,
            'overall_upcoding_rate': (overall_upcoded/total_claims)*100,
            'em_upcoding_rate': (em_upcoded/total_claims)*100,
            'surgical_upcoding_rate': (surgical_upcoded/total_claims)*100,
            'high_risk_claims': len(high_risk_claims),
            'provider_stats': provider_stats
        }
    
    def print_claim_summary(self, claim):
        """Print a formatted summary of a single claim with fraud indicators"""
        print(f"\n{'='*60}")
        print(f"CLAIM SUMMARY - {claim['claim_number']}")
        print(f"{'='*60}")
        print(f"Patient: {claim['patient']['first_name']} {claim['patient']['last_name']} (Age: {claim['patient']['age']})")
        print(f"DOB: {claim['patient']['date_of_birth']}")
        print(f"Provider: {claim['billing_provider']['name']} ({claim['billing_provider']['specialty']})")
        print(f"Service Date: {claim['procedure_lines'][0]['service_date']}")
        print(f"Total Charges: ${claim['total_charges']:.2f}")
        
        print(f"\nFRAUD INDICATORS:")
        print(f"  E&M Upcoding: {'YES' if claim['em_upcoding_flag'] else 'NO'}")
        print(f"  Surgical Upcoding: {'YES' if claim['surgical_upcoding_flag'] else 'NO'}")
        print(f"  Overall Fraud Risk Score: {claim['fraud_risk_score']:.3f}")
        print(f"  Provider Upcoding Tendency: {claim['provider_upcoding_tendency']:.3f}")
        
        if claim['upcoding_details']:
            print(f"  Upcoding Details:")
            for detail in claim['upcoding_details']:
                print(f"    - {detail}")
        
        print(f"\nDiagnoses:")
        for dx in claim['diagnoses']:
            print(f"  {dx['code']}: {dx['description']} (Severity: {dx['severity']})")
        
        print(f"\nProcedures:")
        for line in claim['procedure_lines']:
            appropriate_note = ""
            if 'appropriate_code' in line and line['procedure_code'] != line['appropriate_code']:
                appropriate_note = f" [Should be: {line['appropriate_code']}]"
            print(f"  {line['procedure_code']}: {line['procedure_description']} - ${line['charges']:.2f}{appropriate_note}")


def main():
    """Main function to demonstrate the CMS 1500 upcoding generator"""
    print("CMS 1500 Synthetic Claims Data Generator with Upcoding Detection")
    print("=" * 70)
    
    # Initialize the generator
    generator = CMS1500UpcodingGenerator()
    
    # Get user input
    num_claims_input = input("Enter number of claims to generate (default 100): ") or "100"
    
    try:
        num_claims = int(num_claims_input)
    except ValueError:
        print("Invalid input. Using default of 100 claims.")
        num_claims = 100
    
    # For large datasets, use streaming methods
    if num_claims > 50000:
        print(f"\nLarge dataset detected ({num_claims:,} claims).")
        print("Using memory-efficient streaming generation with fraud detection...")
        
        filename = f'cms1500_claims_with_upcoding_{num_claims}.csv'
        generator.export_to_csv_streaming_with_fraud(num_claims, filename)
        
    else:
        # For smaller datasets, use in-memory generation
        print(f"\nGenerating {num_claims:,} synthetic claims with upcoding patterns...")
        claims = generator.generate_multiple_claims(num_claims)
        
        # Analyze fraud patterns
        fraud_analysis = generator.analyze_fraud_patterns(claims)
        
        # Show sample claims
        if claims:
            print("\nSample Claim (Clean):")
            clean_claims = [c for c in claims if not c['overall_upcoding_flag']]
            if clean_claims:
                generator.print_claim_summary(clean_claims[0])
            
            print("\nSample Claim (Potentially Fraudulent):")
            fraud_claims = [c for c in claims if c['overall_upcoding_flag']]
            if fraud_claims:
                generator.print_claim_summary(fraud_claims[0])
        
        # Export options
        export_choice = input("\nExport format (csv/json/both/none): ").lower()
        
        if export_choice in ['csv', 'both']:
            generator.export_to_csv_with_fraud_flags(claims)
        
        if export_choice in ['json', 'both']:
            generator.export_to_json(claims)
        
        print(f"\nGeneration complete! Created {len(claims):,} synthetic claims with fraud detection.")
    
    print(f"\nKey Features of This Dataset:")
    print(f"- Binary flags for E&M and surgical upcoding")
    print(f"- Fraud risk scores (0.0 to 1.0)")
    print(f"- Provider-specific upcoding tendencies")
    print(f"- Realistic upcoding patterns based on medical coding rules")
    print(f"- Diagnosis severity matching for appropriate complexity")
    print(f"- Detailed upcoding explanations for training purposes")
    
    print(f"\nNote: This synthetic data includes intentional upcoding patterns")
    print(f"for fraud detection training and should only be used for testing.")


def benchmark_performance():
    """Benchmark function to test generation speed with fraud detection"""
    print("Performance Benchmark (With Fraud Detection)")
    print("=" * 50)
    
    generator = CMS1500UpcodingGenerator()
    test_sizes = [100, 1000, 10000]
    
    for size in test_sizes:
        print(f"\nTesting {size:,} claims with fraud detection...")
        start_time = datetime.now()
        
        claims = generator.generate_multiple_claims(size)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Quick fraud analysis
        upcoded_count = sum(1 for claim in claims if claim['overall_upcoding_flag'])
        
        print(f"Generated {len(claims):,} claims in {duration:.2f} seconds")
        print(f"Rate: {len(claims)/duration:.0f} claims/second")
        print(f"Upcoding rate: {(upcoded_count/len(claims))*100:.1f}%")
        
        estimated_time_1m = (1000000 / (len(claims)/duration)) / 60
        print(f"Estimated time for 1M claims: {estimated_time_1m:.1f} minutes")
    
    print("\nBenchmark complete!")


if __name__ == "__main__":
    # Install required packages
    try:
        from faker import Faker
    except ImportError:
        print("Installing required package: faker")
        import subprocess
        subprocess.check_call(["pip", "install", "faker"])
        from faker import Faker
    
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--benchmark':
        benchmark_performance()
    else:
        main()