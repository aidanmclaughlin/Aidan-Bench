import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QSizePolicy, QScrollArea
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QPixmap, QPalette, QColor

class MessageBubble(QLabel):
    def __init__(self, text, is_user=True):
        super().__init__(text)
        self.is_user = is_user
        self.setWordWrap(True)
        self.setFont(QFont("Segoe UI", 12, QFont.Bold, italic=True))
        self.setStyleSheet(self.get_stylesheet())
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.setContentsMargins(10, 5, 10, 5)
        self.adjustSize()

    def get_stylesheet(self):
        if self.is_user:
            # User bubble: Light blue background with thick black border
            return """
                QLabel {
                    background-color: #DCF8C6;
                    border: 2px solid #000000;
                    border-radius: 10px;
                    padding: 10px;
                }
            """
        else:
            # Responder bubble: Light gray background with thick black border
            return """
                QLabel {
                    background-color: #F0F0F0;
                    border: 2px solid #000000;
                    border-radius: 10px;
                    padding: 10px;
                }
            """

class ScoreBox(QLabel):
    def __init__(self, score_type, is_good=True, score=0):
        super().__init__()
        self.score_type = score_type
        self.is_good = is_good
        self.score = score
        self.setFixedSize(90, 30)
        self.setAlignment(Qt.AlignCenter)
        self.setFont(QFont("Calibri", 10, QFont.Bold, italic=True))
        self.setStyleSheet(self.get_stylesheet())
        self.setText(self.get_content())

    def get_stylesheet(self):
        if self.is_good:
            # Green for good score
            return """
                QLabel {
                    background-color: #C8E6C9;
                    border: 1px solid #2E7D32;
                    border-radius: 5px;
                }
            """
        else:
            # Red for bad score
            return """
                QLabel {
                    background-color: #FFCDD2;
                    border: 1px solid #C62828;
                    border-radius: 5px;
                }
            """

    def get_content(self):
        symbol = "✓" if self.is_good else "✗"
        return f"{symbol} {self.score}%"

class AnswerWidget(QWidget):
    def __init__(self, novelty_good=True, novelty_score=0, coherence_good=True, coherence_score=0):
        super().__init__()
        layout = QHBoxLayout()
        layout.setSpacing(20)

        # Novelty
        novelty_label = QLabel("Novelty:")
        novelty_label.setFont(QFont("Calibri", 10, QFont.Bold, italic=True))
        layout.addWidget(novelty_label)

        novelty_box = ScoreBox("Novelty", novelty_good, novelty_score)
        layout.addWidget(novelty_box)

        # Coherence
        coherence_label = QLabel("Coherence:")
        coherence_label.setFont(QFont("Calibri", 10, QFont.Bold, italic=True))
        layout.addWidget(coherence_label)

        coherence_box = ScoreBox("Coherence", coherence_good, coherence_score)
        layout.addWidget(coherence_box)

        layout.addStretch()
        self.setLayout(layout)

class ChatMessage(QWidget):
    def __init__(self, text, is_user=True, is_question=False, profile_pic="images/user.png",
                 novelty_good=True, novelty_score=0, coherence_good=True, coherence_score=0):
        super().__init__()
        main_layout = QHBoxLayout()
        main_layout.setSpacing(10)

        if is_user:
            # User message on the right
            # Stretch to push the message to the right
            main_layout.addStretch()

            # Message bubble
            bubble = MessageBubble(text, is_user=True)
            main_layout.addWidget(bubble)

            # Profile picture
            profile = QLabel()
            pixmap = QPixmap(profile_pic)
            pixmap = pixmap.scaled(40, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            profile.setPixmap(pixmap)
            profile.setFixedSize(40, 40)
            profile.setScaledContents(True)
            main_layout.addWidget(profile)
        else:
            # Responder message on the left
            # Profile picture
            profile = QLabel()
            pixmap = QPixmap(profile_pic)
            pixmap = pixmap.scaled(40, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            profile.setPixmap(pixmap)
            profile.setFixedSize(40, 40)
            profile.setScaledContents(True)
            main_layout.addWidget(profile)

            # Message and scores
            message_layout = QVBoxLayout()
            message_layout.setSpacing(5)

            # Message bubble
            bubble = MessageBubble(text, is_user=False)
            message_layout.addWidget(bubble)

            # Score boxes (only for answers, not for questions)
            if not is_question:
                score_widget = AnswerWidget(novelty_good, novelty_score, coherence_good, coherence_score)
                message_layout.addWidget(score_widget)

            main_layout.addLayout(message_layout)

            # Stretch to push the message to the left
            main_layout.addStretch()

        self.setLayout(main_layout)

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chat Conversation")
        self.setGeometry(100, 100, 800, 600)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Scroll Area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(20)
        scroll_layout.setContentsMargins(10, 10, 10, 10)

        # Question Message (User)
        question_text = "Provide an explanation for Japan's Lost Decades.\n\nProvide an answer you HAVE NOT given previously."
        question_message = ChatMessage(
            text=question_text,
            is_user=True,
            is_question=True,
            profile_pic="images/user.png"
        )
        scroll_layout.addWidget(question_message)

        # Answers (Responder)
        answers = [
            {
                "text": "Japan's Lost Decades were triggered by a combination of excessive debt, asset price collapse, and insufficient monetary policy starting in the early 1990s.",
                "novelty": True,
                "novelty_score": 100,
                "coherence": True,
                "coherence_score": 90
            },
            {
                "text": "We can attribute Japan's Lost Decades to an aging demographic and shrinking productive workforce.",
                "novelty": True,
                "novelty_score": 90,
                "coherence": True,
                "coherence_score": 95
            },
            {
                "text": "Japan's Lost Decades were perpetuated by a combination of economic perils and persistent economic policies that failed to stimulate growth.",
                "novelty": False,
                "novelty_score": 5,
                "coherence": True,
                "coherence_score": 85
            }
        ]

        for ans in answers:
            answer_message = ChatMessage(
                text=ans["text"],
                is_user=False,
                is_question=False,
                profile_pic="images/responder_claude.png",
                novelty_good=ans["novelty"],
                novelty_score=ans.get("novelty_score", 0),
                coherence_good=ans["coherence"],
                coherence_score=ans.get("coherence_score", 0)
            )
            scroll_layout.addWidget(answer_message)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)

        main_layout.addWidget(scroll)
        self.setLayout(main_layout)

def main():
    app = QApplication(sys.argv)

    # Set the main window background to white
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor("#FFFFFF"))  # White background
    palette.setColor(QPalette.WindowText, Qt.black)
    app.setPalette(palette)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
