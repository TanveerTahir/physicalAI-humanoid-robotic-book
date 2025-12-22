# Vibe Coding Handler for Robotics

## Overview
Vibe Coding Handler refers to the practice of maintaining positive team dynamics, effective communication, and collaborative coding practices in robotics development projects. This skill encompasses fostering a productive development environment, managing team interactions, and ensuring smooth collaboration while working on complex robotics systems.

## Key Concepts
- **Team Collaboration**: Effective communication and coordination among team members
- **Code Quality Culture**: Maintaining high standards while keeping the team motivated
- **Conflict Resolution**: Addressing disagreements constructively
- **Knowledge Sharing**: Facilitating learning and expertise distribution
- **Psychological Safety**: Creating an environment where team members feel safe to contribute
- **Remote Collaboration**: Managing distributed teams working on robotics projects

## Essential Vibe Coding Techniques

### 1. Team Communication and Coordination
```python
import asyncio
import logging
from typing import Dict, List, Callable, Any
from dataclasses import dataclass
from datetime import datetime
import json

@dataclass
class TeamMessage:
    sender: str
    message_type: str  # 'question', 'update', 'help', 'celebration', 'concern'
    content: str
    timestamp: datetime
    priority: int = 1  # 1-5 scale
    tags: List[str] = None

class TeamCommunicationManager:
    def __init__(self):
        self.message_queue = asyncio.Queue()
        self.team_members = {}
        self.communication_log = []
        self.sentiment_tracker = {}
        self.response_times = {}

    def add_team_member(self, name: str, role: str, expertise: List[str]):
        """Add team member to communication system"""
        self.team_members[name] = {
            'role': role,
            'expertise': expertise,
            'availability': True,
            'last_activity': datetime.now()
        }

    async def send_message(self, message: TeamMessage):
        """Send message to team communication channel"""
        # Log message
        self.communication_log.append(message)

        # Update sentiment tracking
        self._update_sentiment(message.sender, message.message_type)

        # Route message based on type and priority
        await self._route_message(message)

        print(f"📢 {message.sender}: {message.content}")
        return True

    def _update_sentiment(self, sender: str, message_type: str):
        """Update team sentiment based on message type"""
        if sender not in self.sentiment_tracker:
            self.sentiment_tracker[sender] = {'positive': 0, 'neutral': 0, 'negative': 0}

        if message_type in ['celebration', 'update']:
            self.sentiment_tracker[sender]['positive'] += 1
        elif message_type in ['question', 'help']:
            self.sentiment_tracker[sender]['neutral'] += 1
        elif message_type in ['concern']:
            self.sentiment_tracker[sender]['negative'] += 1

    async def _route_message(self, message: TeamMessage):
        """Route message to appropriate team members"""
        if message.message_type == 'help':
            # Route to experts in relevant area
            await self._route_to_experts(message)
        elif message.message_type == 'question':
            # Route to knowledgeable team members
            await self._route_to_knowledgeable(message)
        elif message.message_type == 'celebration':
            # Broadcast to team
            await self._broadcast_message(message)
        elif message.message_type == 'concern':
            # Route to project lead and relevant stakeholders
            await self._route_to_leadership(message)

    async def _route_to_experts(self, message: TeamMessage):
        """Route help request to relevant experts"""
        # Find team members with relevant expertise
        relevant_experts = [
            name for name, info in self.team_members.items()
            if any(tag in info['expertise'] for tag in (message.tags or []))
        ]

        if relevant_experts:
            for expert in relevant_experts:
                print(f"🔔 Routed to {expert}: {message.content}")
        else:
            print(f"🔔 No specific experts found, broadcasting: {message.content}")
            await self._broadcast_message(message)

    async def _route_to_knowledgeable(self, message: TeamMessage):
        """Route question to knowledgeable members"""
        # For now, broadcast to all team members
        await self._broadcast_message(message)

    async def _broadcast_message(self, message: TeamMessage):
        """Broadcast message to all team members"""
        for member in self.team_members:
            print(f"📢 {member} - {message.content}")

    async def _route_to_leadership(self, message: TeamMessage):
        """Route concerns to leadership"""
        leadership = [
            name for name, info in self.team_members.items()
            if info['role'] in ['lead', 'manager', 'architect']
        ]

        for leader in leadership:
            print(f"⚠️ Leadership Alert - {leader}: {message.content}")

    def get_team_sentiment(self) -> Dict:
        """Get overall team sentiment analysis"""
        total_positive = sum(info['positive'] for info in self.sentiment_tracker.values())
        total_neutral = sum(info['neutral'] for info in self.sentiment_tracker.values())
        total_negative = sum(info['negative'] for info in self.sentiment_tracker.values())

        total_messages = total_positive + total_neutral + total_negative

        if total_messages == 0:
            return {'positive': 0, 'neutral': 0, 'negative': 0}

        return {
            'positive': total_positive / total_messages,
            'neutral': total_neutral / total_messages,
            'negative': total_negative / total_messages
        }

    def celebrate_milestone(self, milestone: str, team_members: List[str]):
        """Celebrate team achievements"""
        celebration_message = TeamMessage(
            sender="TeamBot",
            message_type="celebration",
            content=f"🎉 We've reached milestone: {milestone}! Great work {', '.join(team_members)}! 🚀",
            timestamp=datetime.now(),
            tags=["milestone", "celebration"]
        )
        asyncio.create_task(self.send_message(celebration_message))

class PairProgrammingManager:
    def __init__(self):
        self.active_pairs = {}
        self.pairing_history = []
        self.pairing_preferences = {}

    def suggest_pairs(self, team_members: List[str]) -> List[tuple]:
        """Suggest effective pairing combinations"""
        # Simple algorithm: pair different expertise levels
        pairs = []
        shuffled = team_members.copy()
        import random
        random.shuffle(shuffled)

        for i in range(0, len(shuffled), 2):
            if i + 1 < len(shuffled):
                pairs.append((shuffled[i], shuffled[i + 1]))
            else:
                # If odd number, pair with project lead or most experienced
                pairs.append((shuffled[i], self._get_most_experienced()))

        return pairs

    def _get_most_experienced(self) -> str:
        """Get most experienced team member"""
        # Placeholder implementation
        return "ProjectLead"

    async def start_pairing_session(self, driver: str, navigator: str, task: str):
        """Start a pair programming session"""
        session_id = f"pair_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        session = {
            'id': session_id,
            'driver': driver,
            'navigator': navigator,
            'task': task,
            'start_time': datetime.now(),
            'status': 'active'
        }

        self.active_pairs[session_id] = session

        print(f"👨‍💻 Pair programming session started: {driver} (driver) + {navigator} (navigator) on {task}")

        return session_id

    def end_pairing_session(self, session_id: str):
        """End a pair programming session"""
        if session_id in self.active_pairs:
            session = self.active_pairs[session_id]
            session['end_time'] = datetime.now()
            session['status'] = 'completed'

            self.pairing_history.append(session)
            del self.active_pairs[session_id]

            print(f"✅ Pair programming session completed: {session_id}")
```

### 2. Code Review and Feedback Culture
```python
from enum import Enum
import re

class ReviewQuality(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    ADEQUATE = "adequate"
    NEEDS_IMPROVEMENT = "needs_improvement"
    POOR = "poor"

class CodeReviewManager:
    def __init__(self):
        self.reviews = []
        self.review_guidelines = self._load_guidelines()
        self.feedback_templates = self._load_feedback_templates()

    def _load_guidelines(self) -> Dict:
        """Load code review guidelines"""
        return {
            'functionality': {
                'description': 'Does the code work as intended?',
                'positive_indicators': [
                    'Clear purpose and functionality',
                    'Handles edge cases appropriately',
                    'Follows specifications'
                ],
                'improvement_areas': [
                    'Missing test cases',
                    'Edge case handling',
                    'Error handling'
                ]
            },
            'readability': {
                'description': 'Is the code easy to understand?',
                'positive_indicators': [
                    'Clear variable names',
                    'Good comments where needed',
                    'Logical structure'
                ],
                'improvement_areas': [
                    'Unclear variable names',
                    'Missing documentation',
                    'Complex logic without explanation'
                ]
            },
            'maintainability': {
                'description': 'Is the code easy to maintain?',
                'positive_indicators': [
                    'Modular design',
                    'Follows patterns consistently',
                    'Good error handling'
                ],
                'improvement_areas': [
                    'Tight coupling',
                    'Hardcoded values',
                    'Inconsistent patterns'
                ]
            },
            'performance': {
                'description': 'Does the code perform efficiently?',
                'positive_indicators': [
                    'Efficient algorithms',
                    'Appropriate data structures',
                    'Good resource management'
                ],
                'improvement_areas': [
                    'Inefficient algorithms',
                    'Memory leaks',
                    'Unnecessary computations'
                ]
            }
        }

    def _load_feedback_templates(self) -> Dict:
        """Load positive and constructive feedback templates"""
        return {
            'positive': [
                "Great job on {aspect}! The {specific_thing} is particularly well done.",
                "I really like how you handled {situation}. This approach is {positive_quality}.",
                "Excellent work on {feature}. The {implementation_detail} shows good understanding.",
                "Nice implementation of {pattern}. This makes the code {positive_attribute}."
            ],
            'constructive': [
                "Consider {suggestion} for {reason}. This could improve {benefit}.",
                "What do you think about {alternative_approach} to handle {situation}? {reason}",
                "Perhaps we could {improvement} to make the code more {quality}.",
                "It might be helpful to {suggestion} here because {reason}."
            ],
            'learning_opportunity': [
                "This is a great opportunity to learn about {concept}. Here's a resource: {link}",
                "This scenario demonstrates {principle}. Understanding {aspect} could help here.",
                "This situation relates to {topic}. Exploring {resource} might be beneficial."
            ]
        }

    def conduct_review(self, code: str, author: str, reviewer: str,
                      change_description: str = "") -> Dict:
        """Conduct a code review with positive feedback culture"""
        review = {
            'id': f"review_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'author': author,
            'reviewer': reviewer,
            'change_description': change_description,
            'timestamp': datetime.now(),
            'comments': [],
            'positive_feedback': [],
            'suggestions': [],
            'overall_quality': ReviewQuality.ADEQUATE
        }

        # Analyze code for different aspects
        analysis = self._analyze_code(code)

        # Generate positive feedback
        positive_feedback = self._generate_positive_feedback(analysis, code)
        review['positive_feedback'] = positive_feedback

        # Generate constructive suggestions
        suggestions = self._generate_constructive_suggestions(analysis, code)
        review['suggestions'] = suggestions

        # Calculate overall quality
        review['overall_quality'] = self._calculate_quality(analysis)

        # Add to reviews
        self.reviews.append(review)

        return review

    def _analyze_code(self, code: str) -> Dict:
        """Analyze code for review criteria"""
        analysis = {
            'functionality': self._check_functionality(code),
            'readability': self._check_readability(code),
            'maintainability': self._check_maintainability(code),
            'performance': self._check_performance(code)
        }
        return analysis

    def _check_functionality(self, code: str) -> Dict:
        """Check functionality aspects"""
        # Count function definitions
        function_count = len(re.findall(r'def\s+\w+\s*\(', code))

        # Check for docstrings
        docstring_count = len(re.findall(r'""".*?"""', code, re.DOTALL))

        # Check for error handling
        error_handling = len(re.findall(r'except\s+.*?:', code)) > 0

        return {
            'function_count': function_count,
            'has_docstrings': docstring_count > 0,
            'has_error_handling': error_handling
        }

    def _check_readability(self, code: str) -> Dict:
        """Check readability aspects"""
        # Check variable naming (simple heuristic)
        lines = code.split('\n')
        variables = []
        for line in lines:
            match = re.match(r'^\s*(\w+)\s*=', line)
            if match:
                variables.append(match.group(1))

        readable_vars = sum(1 for var in variables if len(var) > 2 and not var.isdigit())
        total_vars = len(variables)

        return {
            'readable_variables_ratio': readable_vars / max(total_vars, 1),
            'line_count': len(lines)
        }

    def _check_maintainability(self, code: str) -> Dict:
        """Check maintainability aspects"""
        # Check for long functions (heuristic)
        lines = code.split('\n')
        long_functions = 0
        current_function_lines = 0

        for line in lines:
            if line.strip().startswith('def '):
                if current_function_lines > 50:  # More than 50 lines
                    long_functions += 1
                current_function_lines = 0
            current_function_lines += 1

        if current_function_lines > 50:
            long_functions += 1

        return {
            'long_functions_count': long_functions,
            'total_lines': len(lines)
        }

    def _check_performance(self, code: str) -> Dict:
        """Check performance aspects"""
        # Look for potential performance issues
        nested_loops = len(re.findall(r'\s*for.*:\s*\n\s*for', code))
        inefficient_patterns = len(re.findall(r'append|insert|remove', code))  # Simple heuristic

        return {
            'nested_loops': nested_loops,
            'inefficient_patterns': inefficient_patterns
        }

    def _generate_positive_feedback(self, analysis: Dict, code: str) -> List[str]:
        """Generate positive feedback based on analysis"""
        feedback = []

        # Functionality feedback
        if analysis['functionality']['has_docstrings']:
            feedback.append("Great job including docstrings! This makes the code much more understandable.")

        if analysis['functionality']['has_error_handling']:
            feedback.append("Excellent error handling implementation. This shows good defensive programming practices.")

        # Readability feedback
        if analysis['readability']['readable_variables_ratio'] > 0.7:
            feedback.append("The variable naming is very clear and descriptive. This greatly improves code readability.")

        # Maintainability feedback
        if analysis['maintainability']['long_functions_count'] == 0:
            feedback.append("The code is well-structured with appropriately sized functions. This makes it easy to maintain.")

        # Add random positive templates
        import random
        if len(feedback) < 3:
            templates = self.feedback_templates['positive']
            for _ in range(3 - len(feedback)):
                template = random.choice(templates)
                feedback.append(template.format(
                    aspect="the implementation",
                    specific_thing="use of clear functions",
                    positive_quality="very readable",
                    feature="the module",
                    implementation_detail="modular approach",
                    positive_attribute="easy to understand and modify"
                ))

        return feedback

    def _generate_constructive_suggestions(self, analysis: Dict, code: str) -> List[str]:
        """Generate constructive suggestions"""
        suggestions = []

        # Functionality suggestions
        if not analysis['functionality']['has_docstrings']:
            suggestions.append("Consider adding docstrings to document the functions. This helps others understand the purpose and usage.")

        # Performance suggestions
        if analysis['performance']['nested_loops'] > 2:
            suggestions.append("There are several nested loops which might impact performance. Consider if there are more efficient algorithms or data structures that could be used.")

        # Maintainability suggestions
        if analysis['maintainability']['long_functions_count'] > 0:
            suggestions.append("Some functions are quite long. Breaking them into smaller, more focused functions would improve readability and maintainability.")

        # Add constructive templates
        import random
        templates = self.feedback_templates['constructive']
        for _ in range(2):
            template = random.choice(templates)
            suggestions.append(template.format(
                suggestion="adding unit tests",
                reason="ensure code reliability",
                benefit="confidence in future changes",
                alternative_approach="using a more efficient algorithm",
                situation="performance-critical section",
                improvement="adding input validation",
                quality="robust"
            ))

        return suggestions

    def _calculate_quality(self, analysis: Dict) -> ReviewQuality:
        """Calculate overall review quality"""
        score = 0

        # Functionality score
        if analysis['functionality']['has_docstrings']:
            score += 2
        if analysis['functionality']['has_error_handling']:
            score += 2

        # Readability score
        if analysis['readability']['readable_variables_ratio'] > 0.8:
            score += 2
        elif analysis['readability']['readable_variables_ratio'] > 0.5:
            score += 1

        # Maintainability score
        if analysis['maintainability']['long_functions_count'] == 0:
            score += 2
        elif analysis['maintainability']['long_functions_count'] <= 2:
            score += 1

        # Performance score
        if analysis['performance']['nested_loops'] <= 2:
            score += 1
        if analysis['performance']['inefficient_patterns'] <= 3:
            score += 1

        # Map score to quality level
        if score >= 7:
            return ReviewQuality.EXCELLENT
        elif score >= 5:
            return ReviewQuality.GOOD
        elif score >= 3:
            return ReviewQuality.ADEQUATE
        elif score >= 1:
            return ReviewQuality.NEEDS_IMPROVEMENT
        else:
            return ReviewQuality.POOR

    def provide_feedback(self, review: Dict) -> str:
        """Format review feedback in a positive, constructive way"""
        feedback = f"""
🤖 Code Review for {review['author']} by {review['reviewer']}
📅 {review['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
🎯 Quality: {review['overall_quality'].value.title()}

👍 What's Working Well:
"""
        for positive in review['positive_feedback']:
            feedback += f"• {positive}\n"

        if review['suggestions']:
            feedback += "\n💡 Suggestions for Enhancement:\n"
            for suggestion in review['suggestions']:
                feedback += f"• {suggestion}\n"

        feedback += f"\n📋 Summary: Great work on this implementation! {review['change_description']}"
        return feedback
```

### 3. Team Knowledge Sharing and Learning
```python
import hashlib
from typing import Optional

class KnowledgeSharingManager:
    def __init__(self):
        self.knowledge_base = {}
        self.learning_sessions = []
        self.skill_matrices = {}
        self.contribution_tracker = {}

    def create_knowledge_article(self, title: str, content: str, author: str,
                                tags: List[str], category: str = "general") -> str:
        """Create a knowledge article for the team"""
        article_id = hashlib.md5(f"{title}{datetime.now()}".encode()).hexdigest()[:8]

        article = {
            'id': article_id,
            'title': title,
            'content': content,
            'author': author,
            'tags': tags,
            'category': category,
            'created_at': datetime.now(),
            'updated_at': datetime.now(),
            'contributors': [author],
            'views': 0,
            'helpful_votes': 0
        }

        if category not in self.knowledge_base:
            self.knowledge_base[category] = {}

        self.knowledge_base[category][article_id] = article

        # Update contribution tracker
        if author not in self.contribution_tracker:
            self.contribution_tracker[author] = []
        self.contribution_tracker[author].append({
            'type': 'knowledge_article',
            'id': article_id,
            'title': title,
            'timestamp': datetime.now()
        })

        print(f"📚 Knowledge article created: '{title}' by {author}")
        return article_id

    def search_knowledge_base(self, query: str, tags: List[str] = None,
                            category: str = None) -> List[Dict]:
        """Search knowledge base with query and filters"""
        results = []

        for cat, articles in self.knowledge_base.items():
            if category and cat != category:
                continue

            for article_id, article in articles.items():
                # Search in title and content
                search_text = f"{article['title']} {article['content']}".lower()
                query_lower = query.lower()

                matches_query = query_lower in search_text
                matches_tags = not tags or any(tag in article['tags'] for tag in tags)

                if matches_query or matches_tags:
                    results.append(article)

        # Sort by helpful votes and recency
        results.sort(key=lambda x: (x['helpful_votes'], x['updated_at']), reverse=True)
        return results

    def schedule_learning_session(self, topic: str, instructor: str,
                                participants: List[str], duration: int = 60) -> str:
        """Schedule a team learning session"""
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        session = {
            'id': session_id,
            'topic': topic,
            'instructor': instructor,
            'participants': participants,
            'duration': duration,
            'scheduled_time': datetime.now(),
            'status': 'scheduled',
            'materials': [],
            'feedback': []
        }

        self.learning_sessions.append(session)

        print(f"🎓 Learning session scheduled: '{topic}' with {instructor} for {len(participants)} participants")

        # Notify participants
        for participant in participants:
            print(f"Notification: {participant}, you're invited to '{topic}' session")

        return session_id

    def conduct_learning_session(self, session_id: str, materials: List[str] = None):
        """Conduct a learning session"""
        session = next((s for s in self.learning_sessions if s['id'] == session_id), None)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        session['status'] = 'in_progress'
        session['start_time'] = datetime.now()
        session['materials'] = materials or []

        print(f"🎓 Learning session started: {session['topic']}")

        # Simulate session completion
        import time
        time.sleep(2)  # Simulate session duration

        session['status'] = 'completed'
        session['end_time'] = datetime.now()

        print(f"✅ Learning session completed: {session['topic']}")
        return session

    def create_skill_matrix(self, team_members: List[Dict[str, Any]]):
        """Create skill matrix for team"""
        matrix = {}

        for member in team_members:
            name = member['name']
            skills = member['skills']
            expertise_level = member.get('expertise_level', {})

            matrix[name] = {
                'skills': skills,
                'expertise': expertise_level,
                'availability': member.get('availability', 'full'),
                'interests': member.get('interests', [])
            }

        self.skill_matrices = matrix
        return matrix

    def identify_knowledge_gaps(self) -> Dict[str, List[str]]:
        """Identify knowledge gaps in the team"""
        all_skills = set()

        # Collect all required skills
        required_skills = {'ROS2', 'Python', 'C++', 'Control Systems', 'Computer Vision',
                          'Path Planning', 'SLAM', 'Machine Learning', 'Hardware Integration'}

        # Collect team skills
        team_skills = set()
        for member, info in self.skill_matrices.items():
            team_skills.update(info['skills'])

        # Find gaps
        gaps = required_skills - team_skills

        gap_analysis = {
            'missing_skills': list(gaps),
            'skill_coverage': len(team_skills) / len(required_skills) * 100,
            'recommendations': []
        }

        if gaps:
            gap_analysis['recommendations'].append(
                f"Consider training or hiring for these missing skills: {', '.join(gaps)}"
            )

        return gap_analysis

    def track_contributions(self, member: str, contribution_type: str,
                          details: Dict) -> str:
        """Track team member contributions"""
        if member not in self.contribution_tracker:
            self.contribution_tracker[member] = []

        contribution_id = f"contrib_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        contribution = {
            'id': contribution_id,
            'type': contribution_type,
            'details': details,
            'timestamp': datetime.now(),
            'impact_score': details.get('impact_score', 1)
        }

        self.contribution_tracker[member].append(contribution)

        print(f"📊 Contribution tracked: {contribution_type} by {member}")
        return contribution_id

class MentoringManager:
    def __init__(self):
        self.mentorship_pairs = {}
        self.mentoring_sessions = []
        self.progress_tracking = {}

    def establish_mentorship(self, mentor: str, mentee: str,
                           focus_area: str, duration_weeks: int = 12) -> str:
        """Establish a mentorship relationship"""
        pair_id = f"mentorship_{mentor}_{mentee}_{datetime.now().strftime('%Y%m')}"

        pair = {
            'id': pair_id,
            'mentor': mentor,
            'mentee': mentee,
            'focus_area': focus_area,
            'duration_weeks': duration_weeks,
            'start_date': datetime.now(),
            'end_date': datetime.now().replace(weeks=duration_weeks),
            'status': 'active',
            'goals': [],
            'progress': 0.0
        }

        self.mentorship_pairs[pair_id] = pair
        self.progress_tracking[pair_id] = {
            'sessions_attended': 0,
            'goals_achieved': 0,
            'feedback_scores': []
        }

        print(f"🤝 Mentorship established: {mentee} mentored by {mentor} in {focus_area}")
        return pair_id

    def schedule_mentoring_session(self, pair_id: str, scheduled_time: datetime,
                                 agenda: List[str]) -> str:
        """Schedule a mentoring session"""
        session_id = f"session_{pair_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        session = {
            'id': session_id,
            'pair_id': pair_id,
            'scheduled_time': scheduled_time,
            'agenda': agenda,
            'status': 'scheduled',
            'notes': []
        }

        self.mentoring_sessions.append(session)

        pair = self.mentorship_pairs[pair_id]
        print(f"📅 Mentoring session scheduled: {pair['mentee']} with {pair['mentor']}")
        return session_id

    def conduct_mentoring_session(self, session_id: str, notes: str,
                                feedback: int = 5) -> bool:
        """Conduct a mentoring session"""
        session = next((s for s in self.mentoring_sessions if s['id'] == session_id), None)
        if not session:
            return False

        session['status'] = 'completed'
        session['notes'].append({
            'content': notes,
            'timestamp': datetime.now(),
            'feedback_score': feedback
        })

        # Update progress tracking
        pair_id = session['pair_id']
        if pair_id in self.progress_tracking:
            self.progress_tracking[pair_id]['sessions_attended'] += 1
            self.progress_tracking[pair_id]['feedback_scores'].append(feedback)

        print(f"✅ Mentoring session completed: {session_id}")
        return True

    def get_mentoring_progress(self, pair_id: str) -> Dict:
        """Get progress report for a mentorship pair"""
        if pair_id not in self.progress_tracking:
            return {}

        tracking = self.progress_tracking[pair_id]
        pair = self.mentorship_pairs[pair_id]

        avg_feedback = sum(tracking['feedback_scores']) / len(tracking['feedback_scores']) if tracking['feedback_scores'] else 0

        progress_report = {
            'pair_id': pair_id,
            'mentor': pair['mentor'],
            'mentee': pair['mentee'],
            'focus_area': pair['focus_area'],
            'sessions_completed': tracking['sessions_attended'],
            'average_feedback': avg_feedback,
            'estimated_completion': f"{tracking['sessions_attended'] / pair['duration_weeks'] * 100:.1f}%"
        }

        return progress_report
```

### 4. Psychological Safety and Team Dynamics
```python
class PsychologicalSafetyManager:
    def __init__(self):
        self.safety_metrics = {}
        self.trust_building_activities = []
        self.conflict_resolution_history = []
        self.team_feedback_loops = []

    def measure_psychological_safety(self, team_member: str, metrics: Dict) -> Dict:
        """Measure psychological safety for a team member"""
        safety_score = self._calculate_safety_score(metrics)

        self.safety_metrics[team_member] = {
            'score': safety_score,
            'metrics': metrics,
            'last_assessment': datetime.now(),
            'recommendations': self._generate_recommendations(safety_score, metrics)
        }

        return self.safety_metrics[team_member]

    def _calculate_safety_score(self, metrics: Dict) -> float:
        """Calculate psychological safety score (0-10 scale)"""
        # Weighted scoring based on different factors
        weights = {
            'speaking_up_frequency': 0.25,
            'idea_contribution': 0.25,
            'mistake_admission': 0.25,
            'feedback_receptiveness': 0.25
        }

        score = 0.0
        for metric, value in metrics.items():
            if metric in weights:
                # Normalize value to 0-10 scale if needed
                normalized_value = min(10, max(0, value * 10 if value <= 1 else value))
                score += normalized_value * weights[metric]

        return min(10.0, max(0.0, score))

    def _generate_recommendations(self, score: float, metrics: Dict) -> List[str]:
        """Generate recommendations based on safety score"""
        recommendations = []

        if score < 5:
            recommendations.append("Consider implementing more team-building activities to build trust.")
            recommendations.append("Encourage more open communication channels.")
            recommendations.append("Address any team conflicts that may be affecting safety.")

        if metrics.get('speaking_up_frequency', 0) < 0.5:
            recommendations.append("Create more structured opportunities for team members to voice concerns.")

        if metrics.get('mistake_admission', 0) < 0.3:
            recommendations.append("Emphasize that mistakes are learning opportunities, not failures.")
            recommendations.append("Share examples of how mistakes led to improvements.")

        if metrics.get('feedback_receptiveness', 0) < 0.6:
            recommendations.append("Implement regular feedback sessions to normalize the feedback process.")

        return recommendations

    def facilitate_safe_discussion(self, topic: str, participants: List[str],
                                 ground_rules: List[str] = None):
        """Facilitate a discussion with psychological safety in mind"""
        default_rules = [
            "Assume positive intent",
            "Focus on ideas, not people",
            "Build on others' ideas",
            "Ask clarifying questions",
            "Respect different perspectives"
        ]

        rules = ground_rules or default_rules

        print(f"🤝 Safe Discussion Started: {topic}")
        print("Ground Rules:")
        for i, rule in enumerate(rules, 1):
            print(f"{i}. {rule}")

        # Simulate discussion process
        for participant in participants:
            print(f"Listening to {participant}...")
            # In real implementation, this would capture and process input

        print("✅ Discussion completed with psychological safety maintained")

    def resolve_conflict(self, parties: List[str], issue: str,
                        proposed_solution: str) -> bool:
        """Resolve team conflict constructively"""
        conflict_record = {
            'id': f"conflict_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'parties': parties,
            'issue': issue,
            'solution': proposed_solution,
            'resolution_date': datetime.now(),
            'followup_required': True
        }

        print(f"🤝 Conflict Resolution: {issue} between {', '.join(parties)}")
        print(f"Proposed Solution: {proposed_solution}")

        # Steps for resolution
        steps = [
            f"1. Acknowledge the conflict and its impact",
            f"2. Listen to all parties' perspectives",
            f"3. Identify common goals and interests",
            f"4. Brainstorm potential solutions",
            f"5. Agree on the proposed solution",
            f"6. Establish follow-up to ensure resolution"
        ]

        for step in steps:
            print(step)

        self.conflict_resolution_history.append(conflict_record)
        return True

    def implement_feedback_loop(self, team_member: str, feedback_type: str,
                              frequency: str = "weekly") -> str:
        """Implement regular feedback loop for team member"""
        loop_id = f"feedback_{team_member}_{feedback_type}_{frequency}"

        feedback_loop = {
            'id': loop_id,
            'team_member': team_member,
            'type': feedback_type,
            'frequency': frequency,
            'last_feedback': None,
            'next_scheduled': datetime.now(),
            'status': 'active'
        }

        self.team_feedback_loops.append(feedback_loop)

        print(f"🔄 Feedback loop established for {team_member}: {feedback_type} ({frequency})")
        return loop_id

    def create_trust_building_activity(self, activity_name: str,
                                     participants: List[str],
                                     description: str) -> str:
        """Create a trust-building activity"""
        activity_id = f"trust_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        activity = {
            'id': activity_id,
            'name': activity_name,
            'participants': participants,
            'description': description,
            'scheduled_date': datetime.now(),
            'status': 'planned'
        }

        self.trust_building_activities.append(activity)

        print(f"🤝 Trust-building activity planned: {activity_name}")
        return activity_id

class TeamDynamicsAnalyzer:
    def __init__(self):
        self.communication_patterns = {}
        self.collaboration_metrics = {}
        self.engagement_scores = {}

    def analyze_communication_patterns(self, team_member: str,
                                     communication_data: List[Dict]) -> Dict:
        """Analyze communication patterns for a team member"""
        analysis = {
            'response_time_avg': self._calculate_avg_response_time(communication_data),
            'participation_rate': self._calculate_participation_rate(communication_data),
            'positive_tone_ratio': self._calculate_positive_tone_ratio(communication_data),
            'collaboration_frequency': self._calculate_collaboration_frequency(communication_data)
        }

        self.communication_patterns[team_member] = analysis
        return analysis

    def _calculate_avg_response_time(self, data: List[Dict]) -> float:
        """Calculate average response time from communication data"""
        response_times = []
        for item in data:
            if 'response_time' in item:
                response_times.append(item['response_time'])

        return sum(response_times) / len(response_times) if response_times else 0

    def _calculate_participation_rate(self, data: List[Dict]) -> float:
        """Calculate participation rate"""
        total_opportunities = len(data)
        participation_count = sum(1 for item in data if item.get('participated', False))

        return participation_count / total_opportunities if total_opportunities > 0 else 0

    def _calculate_positive_tone_ratio(self, data: List[Dict]) -> float:
        """Calculate ratio of positive communication"""
        positive_count = sum(1 for item in data if item.get('tone') == 'positive')
        total_count = len(data)

        return positive_count / total_count if total_count > 0 else 0

    def _calculate_collaboration_frequency(self, data: List[Dict]) -> int:
        """Calculate collaboration frequency"""
        collaboration_count = sum(1 for item in data if item.get('collaboration_involved', False))
        return collaboration_count

    def generate_team_health_report(self) -> Dict:
        """Generate comprehensive team health report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'overall_health': 'good',
            'communication_health': self._calculate_communication_health(),
            'collaboration_health': self._calculate_collaboration_health(),
            'engagement_health': self._calculate_engagement_health(),
            'recommendations': []
        }

        # Determine overall health based on sub-scores
        avg_health = (
            report['communication_health']['score'] +
            report['collaboration_health']['score'] +
            report['engagement_health']['score']
        ) / 3

        if avg_health >= 8:
            report['overall_health'] = 'excellent'
        elif avg_health >= 6:
            report['overall_health'] = 'good'
        elif avg_health >= 4:
            report['overall_health'] = 'needs_attention'
        else:
            report['overall_health'] = 'poor'

        # Generate recommendations based on health scores
        if report['communication_health']['score'] < 6:
            report['recommendations'].append("Improve team communication through regular check-ins")

        if report['collaboration_health']['score'] < 6:
            report['recommendations'].append("Increase cross-functional collaboration opportunities")

        if report['engagement_health']['score'] < 6:
            report['recommendations'].append("Implement more engaging team activities")

        return report

    def _calculate_communication_health(self) -> Dict:
        """Calculate communication health score"""
        if not self.communication_patterns:
            return {'score': 5, 'details': 'No communication data available'}

        avg_response_time = sum(
            data['response_time_avg'] for data in self.communication_patterns.values()
        ) / len(self.communication_patterns)

        avg_participation = sum(
            data['participation_rate'] for data in self.communication_patterns.values()
        ) / len(self.communication_patterns)

        avg_positive_tone = sum(
            data['positive_tone_ratio'] for data in self.communication_patterns.values()
        ) / len(self.communication_patterns)

        # Score calculation (lower response time is better, so we invert)
        communication_score = (
            (min(10, 10 - avg_response_time) * 0.4) +
            (avg_participation * 10 * 0.3) +
            (avg_positive_tone * 10 * 0.3)
        )

        return {
            'score': communication_score,
            'details': {
                'avg_response_time': avg_response_time,
                'avg_participation_rate': avg_participation,
                'avg_positive_tone_ratio': avg_positive_tone
            }
        }

    def _calculate_collaboration_health(self) -> Dict:
        """Calculate collaboration health score"""
        if not self.communication_patterns:
            return {'score': 5, 'details': 'No collaboration data available'}

        avg_collaboration_freq = sum(
            data['collaboration_frequency'] for data in self.communication_patterns.values()
        ) / len(self.communication_patterns)

        collaboration_score = min(10, avg_collaboration_freq)

        return {
            'score': collaboration_score,
            'details': {
                'avg_collaboration_frequency': avg_collaboration_freq
            }
        }

    def _calculate_engagement_health(self) -> Dict:
        """Calculate engagement health score"""
        if not self.engagement_scores:
            return {'score': 5, 'details': 'No engagement data available'}

        avg_engagement = sum(self.engagement_scores.values()) / len(self.engagement_scores)

        return {
            'score': avg_engagement,
            'details': {
                'engagement_breakdown': self.engagement_scores
            }
        }
```

## Best Practices for Vibe Coding in Robotics

### 1. Remote Collaboration Best Practices
```python
class RemoteCollaborationManager:
    def __init__(self):
        self.remote_sessions = []
        self.virtual_workspace = {}
        self.sync_schedule = {}

    def establish_remote_working_agreement(self, team_members: List[Dict]) -> Dict:
        """Establish remote working agreement for robotics team"""
        agreement = {
            'core_hours': '10:00 AM - 3:00 PM EST',  # Time when all available
            'async_communication': 'Slack for quick questions, email for formal updates',
            'sync_meetings': 'Daily standups at 10 AM, weekly planning on Mondays',
            'code_review_process': 'PRs reviewed within 24 hours during core hours',
            'emergency_contact': 'Slack @here for urgent issues during robot operations',
            'documentation_standard': 'All decisions documented in shared Confluence',
            'pair_programming': 'Remote pairing sessions using VS Code Live Share',
            'virtual_coffee_breaks': 'Weekly informal video calls to maintain team connection'
        }

        print("🏠 Remote Working Agreement Established:")
        for key, value in agreement.items():
            print(f"• {key.replace('_', ' ').title()}: {value}")

        return agreement

    def schedule_synchronous_session(self, session_type: str, participants: List[str],
                                   duration: int = 30, preferred_times: List[str] = None) -> str:
        """Schedule synchronous session for remote team"""
        session_id = f"remote_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        session = {
            'id': session_id,
            'type': session_type,
            'participants': participants,
            'duration': duration,
            'scheduled_time': self._find_best_time(participants, preferred_times),
            'status': 'scheduled',
            'virtual_room': f"room_{session_id}",
            'agenda': []
        }

        self.remote_sessions.append(session)

        print(f"📅 Remote session scheduled: {session_type} for {len(participants)} participants")
        return session_id

    def _find_best_time(self, participants: List[str], preferred_times: List[str]) -> datetime:
        """Find best time for all participants"""
        # For now, use a simple approach - in practice, this would check calendars
        return datetime.now().replace(hour=10, minute=0, second=0, microsecond=0)  # 10 AM

    def create_virtual_workspace(self, workspace_name: str, tools: List[str]) -> str:
        """Create virtual workspace for remote collaboration"""
        workspace_id = f"workspace_{workspace_name}_{datetime.now().strftime('%Y%m')}"

        workspace = {
            'id': workspace_id,
            'name': workspace_name,
            'tools': tools,
            'members': [],
            'channels': {
                'general': f"{workspace_name}-general",
                'code': f"{workspace_name}-code",
                'robot-status': f"{workspace_name}-robot-status",
                'planning': f"{workspace_name}-planning"
            },
            'documents': [],
            'shared_calendar': f"{workspace_name}-calendar"
        }

        self.virtual_workspace[workspace_id] = workspace

        print(f"🌐 Virtual workspace created: {workspace_name}")
        print(f"Available channels: {', '.join(workspace['channels'].values())}")

        return workspace_id
```

### 2. Team Celebration and Recognition
```python
class TeamCelebrationManager:
    def __init__(self):
        self.achievements = []
        self.recognition_system = {}
        self.celebration_calendar = {}

    def recognize_contribution(self, team_member: str, contribution: str,
                            impact: str, peers: List[str] = None) -> str:
        """Recognize team member contribution"""
        recognition_id = f"recognition_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        recognition = {
            'id': recognition_id,
            'team_member': team_member,
            'contribution': contribution,
            'impact': impact,
            'recognized_by': 'Team',
            'peers_involved': peers or [],
            'timestamp': datetime.now(),
            'upvotes': 0
        }

        # Add to recognition system
        if team_member not in self.recognition_system:
            self.recognition_system[team_member] = []
        self.recognition_system[team_member].append(recognition)

        print(f"👏 Recognition for {team_member}: {contribution}")
        print(f"   Impact: {impact}")

        # Celebrate achievement
        self.celebrate_achievement(recognition)

        return recognition_id

    def celebrate_achievement(self, recognition: Dict):
        """Celebrate team achievement"""
        print(f"🎉 Achievement Celebration!")
        print(f"🏆 {recognition['team_member']} achieved: {recognition['contribution']}")
        print(f"✨ Impact: {recognition['impact']}")

        # Send notification to team
        print(f"📢 Team notification: Let's celebrate {recognition['team_member']}'s achievement!")

    def schedule_team_celebration(self, event_name: str, date: datetime,
                                participants: List[str], activities: List[str]) -> str:
        """Schedule team celebration event"""
        event_id = f"celebration_{event_name.replace(' ', '_')}_{date.strftime('%Y%m%d')}"

        event = {
            'id': event_id,
            'name': event_name,
            'date': date,
            'participants': participants,
            'activities': activities,
            'status': 'scheduled',
            'planning_notes': []
        }

        self.celebration_calendar[event_id] = event

        print(f"🎊 Team celebration scheduled: {event_name} on {date.strftime('%Y-%m-%d')}")
        print(f"Participants: {', '.join(participants)}")
        print(f"Activities: {', '.join(activities)}")

        return event_id

    def create_achievement_milestone(self, milestone_name: str,
                                   criteria: str, reward: str) -> str:
        """Create achievement milestone for team"""
        milestone_id = f"milestone_{milestone_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m')}"

        milestone = {
            'id': milestone_id,
            'name': milestone_name,
            'criteria': criteria,
            'reward': reward,
            'achieved_by': [],
            'date_created': datetime.now()
        }

        self.achievements.append(milestone)

        print(f"🏆 Achievement Milestone Created: {milestone_name}")
        print(f"Criteria: {criteria}")
        print(f"Reward: {reward}")

        return milestone_id
```

## Best Practices
- Foster open communication and psychological safety in team interactions
- Implement regular feedback loops and constructive code reviews
- Maintain work-life balance, especially important in demanding robotics projects
- Celebrate achievements and recognize contributions regularly
- Use collaborative tools effectively for remote and distributed teams
- Establish clear ground rules for team interactions and conflict resolution
- Create learning opportunities and mentorship programs
- Maintain documentation and knowledge sharing practices
- Schedule regular team building activities to strengthen relationships
- Monitor team dynamics and address issues proactively