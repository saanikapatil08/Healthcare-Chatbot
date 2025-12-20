"""
Testing and Evaluation Script for Medical Chatbot
Measures performance, accuracy, and reasoning capabilities
"""

import time
import json
import statistics
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you have the chatbot classes imported
# from app import MedicalChatbot


class ChatbotEvaluator:
    """Comprehensive evaluation framework for medical chatbot"""
    
    def __init__(self, chatbot):
        self.chatbot = chatbot
        self.test_results = []
        self.evaluation_metrics = {}
        
    # ========================================================================
    # TEST CASES
    # ========================================================================
    
    def get_test_questions(self) -> List[Dict]:
        """Predefined test questions with expected characteristics"""
        return [
            {
                'question': 'What are the symptoms of diabetes?',
                'category': 'symptoms',
                'complexity': 'simple',
                'expected_keywords': ['blood sugar', 'glucose', 'thirst', 'urination']
            },
            {
                'question': 'How is hypertension diagnosed?',
                'category': 'diagnosis',
                'complexity': 'medium',
                'expected_keywords': ['blood pressure', 'measurement', 'mmHg']
            },
            {
                'question': 'What is the difference between Type 1 and Type 2 diabetes?',
                'category': 'comparison',
                'complexity': 'complex',
                'expected_keywords': ['insulin', 'pancreas', 'autoimmune', 'resistance']
            },
            {
                'question': 'What are the risk factors for heart disease?',
                'category': 'risk_factors',
                'complexity': 'medium',
                'expected_keywords': ['cholesterol', 'smoking', 'obesity', 'age']
            },
            {
                'question': 'Explain the mechanism of action of beta-blockers',
                'category': 'mechanism',
                'complexity': 'complex',
                'expected_keywords': ['heart rate', 'blood pressure', 'adrenaline', 'receptors']
            },
            {
                'question': 'What are common causes of headache?',
                'category': 'causes',
                'complexity': 'simple',
                'expected_keywords': ['tension', 'migraine', 'stress', 'dehydration']
            },
            {
                'question': 'How does the immune system respond to infection?',
                'category': 'mechanism',
                'complexity': 'complex',
                'expected_keywords': ['antibodies', 'white blood cells', 'pathogen', 'immune']
            },
            {
                'question': 'What lifestyle changes help manage diabetes?',
                'category': 'treatment',
                'complexity': 'medium',
                'expected_keywords': ['diet', 'exercise', 'weight', 'monitoring']
            }
        ]
    
    # ========================================================================
    # PERFORMANCE METRICS
    # ========================================================================
    
    def measure_response_time(self, question: str) -> Tuple[str, float]:
        """Measure response time for a single query"""
        start_time = time.time()
        result = self.chatbot.query(question)
        response_time = time.time() - start_time
        return result['answer'], response_time
    
    def evaluate_response_latency(self) -> Dict:
        """Evaluate response latency across multiple queries"""
        print("\nEvaluating Response Latency...")
        
        test_questions = self.get_test_questions()
        response_times = []
        
        for i, test_case in enumerate(test_questions, 1):
            print(f"  Testing query {i}/{len(test_questions)}...", end=" ")
            _, response_time = self.measure_response_time(test_case['question'])
            response_times.append(response_time)
            print(f"{response_time:.2f}s")
        
        metrics = {
            'min_time': min(response_times),
            'max_time': max(response_times),
            'avg_time': statistics.mean(response_times),
            'median_time': statistics.median(response_times),
            'std_dev': statistics.stdev(response_times) if len(response_times) > 1 else 0,
            'all_times': response_times
        }
        
        print(f"\nLatency Evaluation Complete:")
        print(f"  Min: {metrics['min_time']:.2f}s")
        print(f"  Max: {metrics['max_time']:.2f}s")
        print(f"  Avg: {metrics['avg_time']:.2f}s")
        print(f"  Median: {metrics['median_time']:.2f}s")
        
        return metrics
    
    # ========================================================================
    # ACCURACY & QUALITY METRICS
    # ========================================================================
    
    def evaluate_keyword_coverage(self, answer: str, expected_keywords: List[str]) -> float:
        """Check if answer contains expected medical keywords"""
        answer_lower = answer.lower()
        found_keywords = sum(1 for keyword in expected_keywords 
                           if keyword.lower() in answer_lower)
        coverage = found_keywords / len(expected_keywords) if expected_keywords else 0
        return coverage
    
    def evaluate_answer_length(self, answer: str) -> Dict:
        """Evaluate answer completeness based on length"""
        words = answer.split()
        chars = len(answer)
        
        # Quality heuristics
        is_too_short = len(words) < 20
        is_too_long = len(words) > 500
        is_adequate = 50 <= len(words) <= 300
        
        return {
            'word_count': len(words),
            'char_count': chars,
            'too_short': is_too_short,
            'too_long': is_too_long,
            'adequate_length': is_adequate
        }
    
    def evaluate_source_relevance(self, sources: List[str]) -> float:
        """Evaluate if sources are provided and relevant"""
        if not sources:
            return 0.0
        
        # Higher score for multiple unique sources
        unique_sources = len(set(sources))
        relevance_score = min(unique_sources / 3.0, 1.0)
        return relevance_score
    
    def evaluate_response_quality(self, test_case: Dict, result: Dict) -> Dict:
        """Comprehensive quality evaluation"""
        answer = result['answer']
        sources = result['sources']
        
        # Keyword coverage
        keyword_coverage = self.evaluate_keyword_coverage(
            answer, 
            test_case.get('expected_keywords', [])
        )
        
        # Length evaluation
        length_metrics = self.evaluate_answer_length(answer)
        
        # Source relevance
        source_relevance = self.evaluate_source_relevance(sources)
        
        # Overall quality score
        quality_score = (
            keyword_coverage * 0.4 +
            (1.0 if length_metrics['adequate_length'] else 0.5) * 0.3 +
            source_relevance * 0.3
        )
        
        return {
            'keyword_coverage': keyword_coverage,
            'length_metrics': length_metrics,
            'source_relevance': source_relevance,
            'quality_score': quality_score
        }
    
    # ========================================================================
    # REASONING EVALUATION
    # ========================================================================
    
    def evaluate_reasoning_capabilities(self) -> Dict:
        """Evaluate chatbot's reasoning across different question types"""
        print("\nEvaluating Reasoning Capabilities...")
        
        test_questions = self.get_test_questions()
        reasoning_scores = {
            'simple': [],
            'medium': [],
            'complex': []
        }
        
        for test_case in test_questions:
            print(f"  Testing: {test_case['category']} ({test_case['complexity']})...", end=" ")
            
            result = self.chatbot.query(test_case['question'])
            quality = self.evaluate_response_quality(test_case, result)
            
            complexity = test_case['complexity']
            reasoning_scores[complexity].append(quality['quality_score'])
            
            print(f"Score: {quality['quality_score']:.2f}")
        
        # Calculate averages
        avg_scores = {
            complexity: statistics.mean(scores) if scores else 0
            for complexity, scores in reasoning_scores.items()
        }
        
        print(f"\nReasoning Evaluation:")
        print(f"  Simple questions: {avg_scores['simple']:.2f}")
        print(f"  Medium questions: {avg_scores['medium']:.2f}")
        print(f"  Complex questions: {avg_scores['complex']:.2f}")
        
        return {
            'scores_by_complexity': reasoning_scores,
            'average_scores': avg_scores
        }
    
    # ========================================================================
    # COMPREHENSIVE EVALUATION
    # ========================================================================
    
    def run_full_evaluation(self) -> Dict:
        """Run comprehensive evaluation suite"""
        print("\n" + "="*70)
        print("MEDICAL CHATBOT EVALUATION SUITE")
        print("="*70)
        
        # 1. Response Latency
        latency_metrics = self.evaluate_response_latency()
        
        # 2. Reasoning Capabilities
        reasoning_metrics = self.evaluate_reasoning_capabilities()
        
        # 3. Detailed Test Cases
        print("\nRunning Detailed Test Cases...")
        test_questions = self.get_test_questions()
        detailed_results = []
        
        for i, test_case in enumerate(test_questions, 1):
            print(f"\n  Test Case {i}/{len(test_questions)}: {test_case['category']}")
            
            start_time = time.time()
            result = self.chatbot.query(test_case['question'])
            response_time = time.time() - start_time
            
            quality = self.evaluate_response_quality(test_case, result)
            
            detailed_result = {
                'test_case': test_case,
                'result': result,
                'response_time': response_time,
                'quality_metrics': quality
            }
            detailed_results.append(detailed_result)
            
            print(f"    Response Time: {response_time:.2f}s")
            print(f"    Quality Score: {quality['quality_score']:.2f}")
            print(f"    Keyword Coverage: {quality['keyword_coverage']:.2%}")
        
        # Compile final report
        evaluation_report = {
            'timestamp': datetime.now().isoformat(),
            'latency_metrics': latency_metrics,
            'reasoning_metrics': reasoning_metrics,
            'detailed_results': detailed_results,
            'summary': self._generate_summary(
                latency_metrics, 
                reasoning_metrics, 
                detailed_results
            )
        }
        
        self.evaluation_metrics = evaluation_report
        return evaluation_report
    
    def _generate_summary(self, latency, reasoning, detailed) -> Dict:
        """Generate evaluation summary"""
        all_quality_scores = [r['quality_metrics']['quality_score'] 
                            for r in detailed]
        
        return {
            'total_tests': len(detailed),
            'avg_response_time': latency['avg_time'],
            'avg_quality_score': statistics.mean(all_quality_scores),
            'overall_performance': self._categorize_performance(
                latency['avg_time'],
                statistics.mean(all_quality_scores)
            )
        }
    
    def _categorize_performance(self, avg_time: float, avg_quality: float) -> str:
        """Categorize overall performance"""
        if avg_time < 5 and avg_quality > 0.8:
            return "Excellent"
        elif avg_time < 10 and avg_quality > 0.6:
            return "Good"
        elif avg_time < 15 and avg_quality > 0.5:
            return "Fair"
        else:
            return "Needs Improvement"
    
    # ========================================================================
    # REPORTING & VISUALIZATION
    # ========================================================================
    
    def save_report(self, filename: str = "evaluation_report.json"):
        """Save evaluation report to JSON file"""
        output_path = Path("reports") / filename
        output_path.parent.mkdir(exist_ok=True)
        
        # Convert to JSON-serializable format
        report = self._prepare_report_for_json(self.evaluation_metrics)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nReport saved to: {output_path}")
    
    def _prepare_report_for_json(self, report: Dict) -> Dict:
        """Prepare report for JSON serialization"""
        # Remove non-serializable objects
        json_report = report.copy()
        if 'detailed_results' in json_report:
            for result in json_report['detailed_results']:
                if 'result' in result and 'source_documents' in result['result']:
                    result['result']['source_documents'] = [
                        str(doc) for doc in result['result']['source_documents']
                    ]
        return json_report
    
    def print_summary_report(self):
        """Print formatted summary report"""
        if not self.evaluation_metrics:
            print("No evaluation data available. Run evaluation first.")
            return
        
        summary = self.evaluation_metrics['summary']
        
        print("\n" + "="*70)
        print("EVALUATION SUMMARY REPORT")
        print("="*70)
        print(f"\nTotal Tests: {summary['total_tests']}")
        print(f"Average Response Time: {summary['avg_response_time']:.2f}s")
        print(f"Average Quality Score: {summary['avg_quality_score']:.2%}")
        print(f"Overall Performance: {summary['overall_performance']}")
        print("\n" + "="*70)
    
    def plot_evaluation_results(self):
        """Create visualization of evaluation results"""
        if not self.evaluation_metrics:
            print("No evaluation data available.")
            return
        
        detailed = self.evaluation_metrics['detailed_results']
        
        # Extract data
        categories = [r['test_case']['category'] for r in detailed]
        response_times = [r['response_time'] for r in detailed]
        quality_scores = [r['quality_metrics']['quality_score'] for r in detailed]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Medical Chatbot Evaluation Results', fontsize=16)
        
        # 1. Response Time by Category
        axes[0, 0].bar(range(len(categories)), response_times, color='steelblue')
        axes[0, 0].set_xticks(range(len(categories)))
        axes[0, 0].set_xticklabels(categories, rotation=45, ha='right')
        axes[0, 0].set_ylabel('Response Time (s)')
        axes[0, 0].set_title('Response Time by Question Category')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # 2. Quality Score by Category
        axes[0, 1].bar(range(len(categories)), quality_scores, color='seagreen')
        axes[0, 1].set_xticks(range(len(categories)))
        axes[0, 1].set_xticklabels(categories, rotation=45, ha='right')
        axes[0, 1].set_ylabel('Quality Score')
        axes[0, 1].set_title('Quality Score by Question Category')
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # 3. Response Time Distribution
        axes[1, 0].hist(response_times, bins=10, color='coral', edgecolor='black')
        axes[1, 0].set_xlabel('Response Time (s)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Response Time Distribution')
        axes[1, 0].axvline(statistics.mean(response_times), 
                          color='red', linestyle='--', label='Mean')
        axes[1, 0].legend()
        
        # 4. Quality vs Response Time Scatter
        axes[1, 1].scatter(response_times, quality_scores, 
                          c=quality_scores, cmap='RdYlGn', s=100, alpha=0.6)
        axes[1, 1].set_xlabel('Response Time (s)')
        axes[1, 1].set_ylabel('Quality Score')
        axes[1, 1].set_title('Quality vs Response Time')
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_path = Path("reports") / "evaluation_plots.png"
        output_path.parent.mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nPlots saved to: {output_path}")
        
        plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main evaluation execution"""
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║   Medical Chatbot - Evaluation & Testing Suite           ║
    ║   Measures Performance, Accuracy, and Reasoning          ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize chatbot (you need to import and initialize your chatbot here)
    # from app import MedicalChatbot
    # chatbot = MedicalChatbot(model_path, db_path)
    # chatbot.initialize()
    
    # For demonstration, assuming chatbot is already initialized
    # chatbot = ...  # Your initialized chatbot
    
    print("\nNote: Make sure to initialize your chatbot before running evaluation")
    print("Example:")
    print("  from app import MedicalChatbot")
    print("  chatbot = MedicalChatbot('models/...', 'vectorstore/...')")
    print("  chatbot.initialize()")
    print("  evaluator = ChatbotEvaluator(chatbot)")
    print("  results = evaluator.run_full_evaluation()")
    
    # evaluator = ChatbotEvaluator(chatbot)
    
    # Run full evaluation
    # results = evaluator.run_full_evaluation()
    
    # Save results
    # evaluator.save_report()
    
    # Print summary
    # evaluator.print_summary_report()
    
    # Create visualizations
    # evaluator.plot_evaluation_results()


if __name__ == "__main__":
    main()

