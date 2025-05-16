import sys
import os
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QComboBox, QPushButton, 
                            QFrame, QSpacerItem, QSizePolicy, QTextEdit, 
                            QDoubleSpinBox, QGridLayout, QScrollArea, QGraphicsDropShadowEffect)
from PyQt5.QtCore import Qt, QSize, QTimer, QPropertyAnimation, QEasingCurve, QPoint, QRect
from PyQt5.QtGui import QFont, QIcon, QPixmap, QColor, QPalette, QLinearGradient, QPainter, QPainterPath, QRadialGradient

class ModernMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Student Habits and Performance Analysis")
        self.setMinimumSize(1000, 700)
        
        # Set the application style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #121212;
            }
            QWidget {
                font-family: 'Segoe UI', sans-serif;
            }
            QLabel {
                color: white;
            }
            QComboBox {
                background-color: rgba(30, 30, 30, 0.8);
                color: white;
                border: 1px solid #3d5afe;
                border-radius: 6px;
                padding: 8px;
                min-height: 30px;
                selection-background-color: #3d5afe;
            }
            QComboBox:hover {
                border: 1px solid #536dfe;
                background-color: rgba(40, 40, 40, 0.8);
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: url(down_arrow.png);
                width: 12px;
                height: 12px;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3d5afe, stop:1 #1a237e);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-weight: bold;
                min-height: 35px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #536dfe, stop:1 #283593);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1a237e, stop:1 #0d47a1);
            }
            QTextEdit {
                background-color: rgba(30, 30, 30, 0.8);
                color: white;
                border: 1px solid #3d5afe;
                border-radius: 10px;
                padding: 15px;
                selection-background-color: #3d5afe;
            }
            QDoubleSpinBox {
                background-color: rgba(30, 30, 30, 0.8);
                color: white;
                border: 1px solid #3d5afe;
                border-radius: 6px;
                padding: 8px;
                min-height: 30px;
            }
            QFrame#card {
                background-color: rgba(30, 30, 30, 0.8);
                border-radius: 12px;
                border: 1px solid #3d5afe;
                padding-right: 20px;  /* Increased padding from 10px to 20px */
            }
            QFrame#header {
                background-color: rgba(30, 30, 30, 0.8);
                border-radius: 12px;
                border: 1px solid #3d5afe;
            }
            QFrame#footer {
                background-color: rgba(30, 30, 30, 0.8);
                border-radius: 12px;
                border: 1px solid #3d5afe;
            }
            QScrollBar:vertical {
                background-color: rgba(30, 30, 30, 0.8);
                width: 10px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background-color: #3d5afe;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        
        # Create central widget and main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        self.main_layout.setSpacing(20)
        
        # Create header
        self.create_header()
        
        # Create content area with left sidebar and right content
        self.content_layout = QHBoxLayout()
        self.content_layout.setSpacing(30)  # Further increased spacing between sidebar and main content
        
        # Create left sidebar with cards
        self.left_sidebar = QWidget()
        self.left_sidebar.setMaximumWidth(400)
        self.left_sidebar_layout = QVBoxLayout(self.left_sidebar)
        self.left_sidebar_layout.setSpacing(15)
        
        # Create scroll area for the sidebar
        self.sidebar_scroll = QScrollArea()
        self.sidebar_scroll.setWidgetResizable(True)
        self.sidebar_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.sidebar_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.sidebar_scroll.setStyleSheet("""
            QScrollArea {
                background-color: rgba(30, 30, 30, 0.8);
                border: 1px solid #3d5afe;
                border-radius: 12px;
                padding: 5px 0px 5px 5px;  /* No right padding to let scrollbar sit at edge */
            }
            QScrollArea > QWidget > QWidget {
                background-color: transparent;
            }
            QScrollBar:vertical {
                background-color: rgba(30, 30, 30, 0.8);
                width: 22px;  /* Increased width from 16px to 22px to make it thicker */
                margin: 0;  /* No margins to position right at the edge */
                border-top-right-radius: 12px;
                border-bottom-right-radius: 12px;
            }
            QScrollBar::handle:vertical {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3d5afe, stop:1 #536dfe);
                min-height: 30px;
                border-radius: 8px;  /* Increased border radius for a more rounded look */
                margin: 4px 2px;  /* Adjusted margins to center the handle */
                border: 1px solid #1a237e;
            }
            QScrollBar::handle:vertical:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #536dfe, stop:1 #8c9eff);
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: rgba(30, 30, 30, 0.4);
                border-radius: 0;  /* No border radius for scroll pages */
            }
        """)
        
        # Apply shadow effect to scroll area
        scroll_shadow = QGraphicsDropShadowEffect()
        scroll_shadow.setBlurRadius(15)
        scroll_shadow.setColor(QColor(0, 0, 0, 60))
        scroll_shadow.setOffset(0, 3)
        self.sidebar_scroll.setGraphicsEffect(scroll_shadow)
        
        # Create a container widget for the cards
        self.cards_container = QWidget()
        self.cards_layout = QVBoxLayout(self.cards_container)
        self.cards_layout.setSpacing(15)
        self.cards_layout.setContentsMargins(8, 8, 40, 8)  # Increased right margin to 40px to keep content away from scrollbar
        
        # Create cards for each analysis type
        self.create_pie_chart_card()
        self.create_regression_card()
        self.create_eda_card()
        self.create_distribution_card()
        self.create_confidence_card()
        self.create_data_table_card()
        
        # Add spacer to push cards to the top
        self.cards_layout.addStretch()
        
        # Set the cards container as the scroll area's widget
        self.sidebar_scroll.setWidget(self.cards_container)
        
        # Create right content area
        self.right_content = QWidget()
        self.right_content_layout = QVBoxLayout(self.right_content)
        
        # Create output area
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setMinimumHeight(500)
        self.output_text.setPlaceholderText("Results will appear here")
        
        self.right_content_layout.addWidget(self.output_text)
        
        # Add left sidebar and right content to content layout
        self.content_layout.addWidget(self.sidebar_scroll)
        self.content_layout.addWidget(self.right_content, 1)  # 1 is the stretch factor
        
        # Add content layout to main layout
        self.main_layout.addLayout(self.content_layout)
        
        # Create footer
        self.create_footer()
        
        # Create loading overlay (hidden by default)
        self.loading_overlay = QWidget(self)
        self.loading_overlay.setGeometry(0, 0, self.width(), self.height())
        self.loading_overlay.setStyleSheet("background-color: rgba(0, 0, 0, 0.7);")
        self.loading_overlay.hide()
        
        self.loading_label = QLabel("Processing...", self.loading_overlay)
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.loading_label.setStyleSheet("color: white; font-size: 24px; font-weight: bold;")
        self.loading_label.setGeometry(0, 0, self.width(), self.height())
        
        # Create a timer for the loading animation
        self.loading_timer = QTimer()
        self.loading_timer.timeout.connect(self.update_loading_animation)
        self.loading_dots = 0
        
        # Create a background image with overlay
        self.set_background()
        
        # Apply shadow effects to cards
        self.apply_shadow_effects()
        
        # Apply hover effects to buttons
        self.apply_button_hover_effects()
    
    def set_background(self):
        # Create a more sophisticated background with gradient
        palette = self.palette()
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, QColor("#121212"))
        gradient.setColorAt(0.5, QColor("#1a1a1a"))
        gradient.setColorAt(1, QColor("#0d47a1"))
        palette.setBrush(QPalette.Window, gradient)
        self.setPalette(palette)
    
    def apply_shadow_effects(self):
        # Apply shadow effects to all cards
        for i in range(self.cards_layout.count()):
            widget = self.cards_layout.itemAt(i).widget()
            if isinstance(widget, QFrame):
                shadow = QGraphicsDropShadowEffect()
                shadow.setBlurRadius(15)
                shadow.setColor(QColor(0, 0, 0, 60))
                shadow.setOffset(0, 3)
                widget.setGraphicsEffect(shadow)
    
    def apply_button_hover_effects(self):
        # Apply hover effects to all buttons
        for i in range(self.cards_layout.count()):
            widget = self.cards_layout.itemAt(i).widget()
            if isinstance(widget, QFrame):
                for child in widget.findChildren(QPushButton):
                    # Store the original effect
                    child.setProperty("original_effect", child.graphicsEffect())
                    
                    # Create a custom enter/leave event handler
                    child.enterEvent = lambda event, btn=child: self.button_enter_event(event, btn)
                    child.leaveEvent = lambda event, btn=child: self.button_leave_event(event, btn)
    
    def button_enter_event(self, event, button):
        # Create a new shadow effect on hover
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(10)
        shadow.setColor(QColor(61, 90, 254, 150))  # #3d5afe with alpha
        shadow.setOffset(0, 0)
        
        # Apply the shadow effect
        button.setGraphicsEffect(shadow)
        
        # Call the parent class's enterEvent
        QPushButton.enterEvent(button, event)
    
    def button_leave_event(self, event, button):
        # Restore original effect (or none) on leave
        original_effect = button.property("original_effect")
        button.setGraphicsEffect(original_effect)
        
        # Call the parent class's leaveEvent
        QPushButton.leaveEvent(button, event)
    
    def create_header(self):
        header_frame = QFrame()
        header_frame.setObjectName("header")
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(15, 10, 15, 10)
        
        # Create a container for the logo and title with horizontal layout
        logo_title_container = QWidget()
        logo_title_layout = QHBoxLayout(logo_title_container)
        logo_title_layout.setContentsMargins(0, 0, 0, 0)
        logo_title_layout.setSpacing(15)  # Space between logo and title
        
        # Add logo - using the existing logo.png file
        logo_label = QLabel()
        pixmap = QPixmap("logo.png")
        # Scale the logo to an appropriate size
        pixmap = pixmap.scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        logo_label.setPixmap(pixmap)
        logo_label.setAlignment(Qt.AlignCenter)
        logo_title_layout.addWidget(logo_label)
        
        # Center-aligned title
        title = QLabel("Analysis of Student Habits and Performance")
        title_font = QFont("Segoe UI", 24, QFont.Bold)
        title.setFont(title_font)
        title.setStyleSheet("color: white;")
        title.setAlignment(Qt.AlignVCenter)  # Align vertically center to match logo
        logo_title_layout.addWidget(title)
        
        # Add the logo and title container to the layout with stretch on both sides to center it
        header_layout.addStretch()
        header_layout.addWidget(logo_title_container)
        header_layout.addStretch()
        
        # Apply shadow effect to header
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 60))
        shadow.setOffset(0, 3)
        header_frame.setGraphicsEffect(shadow)
        
        self.main_layout.addWidget(header_frame)
    
    def create_pie_chart_card(self):
        # Create a card for pie chart options
        pie_card = QFrame()
        pie_card.setObjectName("card")
        pie_card_layout = QVBoxLayout(pie_card)
        pie_card_layout.setContentsMargins(15, 15, 35, 15)  # Increased right padding to 35px
        
        # Add a title
        title = QLabel("Categorical Analysis")
        title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        pie_card_layout.addWidget(title)
        
        # Add a combo box for variable selection
        var_layout = QHBoxLayout()
        var_label = QLabel("Variable:")
        var_label.setFixedWidth(80)
        self.comboPie = QComboBox()
        
        # Add categorical variables to the combo box
        self.comboPie.addItems(["gender", "part_time_job", "diet_quality", 
                              "parental_education_level", "internet_quality", 
                              "extracurricular_participation"])
        
        var_layout.addWidget(var_label)
        var_layout.addWidget(self.comboPie)
        pie_card_layout.addLayout(var_layout)
        
        # Add a button to generate the chart
        self.SearchPie = QPushButton("Generate Pie Chart")
        pie_card_layout.addWidget(self.SearchPie)
        
        # Add the card to the cards layout
        self.cards_layout.addWidget(pie_card)
    
    def create_regression_card(self):
        # Create a card for regression analysis options
        regression_card = QFrame()
        regression_card.setObjectName("card")
        regression_card_layout = QVBoxLayout(regression_card)
        regression_card_layout.setContentsMargins(15, 15, 35, 15)  # Increased right padding to 35px
        
        # Add a title
        title = QLabel("Regression Analysis")
        title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        regression_card_layout.addWidget(title)
        
        # Add a combo box for regression type selection
        type_layout = QHBoxLayout()
        type_label = QLabel("Predict:")
        type_label.setFixedWidth(80)
        self.comboPie_Regression = QComboBox()
        
        # Add regression types to the combo box
        self.comboPie_Regression.addItems(["Predict exam score", "Predict attendance"])
        
        type_layout.addWidget(type_label)
        type_layout.addWidget(self.comboPie_Regression)
        regression_card_layout.addLayout(type_layout)
        
        # Add a button to generate the regression model
        self.SearchPie_Regression = QPushButton("Run Regression")
        regression_card_layout.addWidget(self.SearchPie_Regression)
        
        # Add a spacer
        regression_card_layout.addSpacing(10)
        
        # Add a separator line
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background-color: #3d5afe;")
        regression_card_layout.addWidget(line)
        
        # Add a title for prediction
        prediction_title = QLabel("Make Prediction")
        prediction_title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        prediction_title.setAlignment(Qt.AlignCenter)
        regression_card_layout.addWidget(prediction_title)
        
        # Add a combo box for prediction type selection
        pred_type_layout = QHBoxLayout()
        pred_type_label = QLabel("Predict:")
        pred_type_label.setFixedWidth(80)
        self.comboPie_Regression_predict = QComboBox()
        
        # Add prediction types to the combo box
        self.comboPie_Regression_predict.addItems(["Predict exam score", "Predict attendance"])
        
        pred_type_layout.addWidget(pred_type_label)
        pred_type_layout.addWidget(self.comboPie_Regression_predict)
        regression_card_layout.addLayout(pred_type_layout)
        
        # Add a spin box for the input value
        input_layout = QHBoxLayout()
        input_label = QLabel("Hours:")
        input_label.setFixedWidth(80)
        self.spinBox_predict = QDoubleSpinBox()
        self.spinBox_predict.setRange(0, 10)  # Set range for study/sleep hours
        self.spinBox_predict.setValue(4)  # Set default value
        self.spinBox_predict.setSingleStep(0.5)  # Set step size
        self.spinBox_predict.setDecimals(1)  # Show one decimal place
        
        input_layout.addWidget(input_label)
        input_layout.addWidget(self.spinBox_predict)
        regression_card_layout.addLayout(input_layout)
        
        # Add a button to generate the prediction
        self.SearchPie_Regression_predict = QPushButton("Make Prediction")
        regression_card_layout.addWidget(self.SearchPie_Regression_predict)
        
        # Add the card to the cards layout
        self.cards_layout.addWidget(regression_card)
    
    def create_eda_card(self):
        # Create a card for exploratory data analysis options
        eda_card = QFrame()
        eda_card.setObjectName("card")
        eda_card_layout = QVBoxLayout(eda_card)
        eda_card_layout.setContentsMargins(15, 15, 35, 15)  # Increased right padding to 35px
        
        # Add a title
        title = QLabel("EDA")
        title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        eda_card_layout.addWidget(title)
        
        # Add a combo box for variable selection
        var_layout = QHBoxLayout()
        var_label = QLabel("Variable:")
        var_label.setFixedWidth(80)
        self.comboPie_EDA = QComboBox()
        
        # Add numerical variables to the combo box
        self.comboPie_EDA.addItems(["study_hours_per_day", "social_media_hours", "netflix_hours", 
                                   "attendance_percentage", "sleep_hours", "exercise_frequency", 
                                   "mental_health_rating", "exam_score", "age"])
        
        var_layout.addWidget(var_label)
        var_layout.addWidget(self.comboPie_EDA)
        eda_card_layout.addLayout(var_layout)
        
        # Add a button to generate the EDA
        self.SearchPie_EDA = QPushButton("Run EDA")
        eda_card_layout.addWidget(self.SearchPie_EDA)
        
        # Add the card to the cards layout
        self.cards_layout.addWidget(eda_card)
    
    def create_distribution_card(self):
        # Create a card for probability distribution options
        dist_card = QFrame()
        dist_card.setObjectName("card")
        dist_card_layout = QVBoxLayout(dist_card)
        dist_card_layout.setContentsMargins(15, 15, 35, 15)  # Increased right padding to 35px
        
        # Add a title
        title = QLabel("Distributions")
        title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        dist_card_layout.addWidget(title)
        
        # Add a combo box for distribution type selection
        dist_layout = QHBoxLayout()
        dist_label = QLabel("Type:")
        dist_label.setFixedWidth(80)
        self.comboPie_distribution = QComboBox()
        
        # Add distribution types to the combo box
        self.comboPie_distribution.addItems([
            "Normal Distribution", 
            "Binomial Distribution", 
            "Poisson Distribution",
            "Uniform Distribution"
        ])
        
        dist_layout.addWidget(dist_label)
        dist_layout.addWidget(self.comboPie_distribution)
        dist_card_layout.addLayout(dist_layout)
        
        # Add a button to generate the distribution
        self.SearchPie_distribution = QPushButton("Generate Distribution")
        dist_card_layout.addWidget(self.SearchPie_distribution)
        
        # Add the card to the cards layout
        self.cards_layout.addWidget(dist_card)
    
    def create_confidence_card(self):
        # Create a card for confidence interval options
        self.confidence_card = QFrame()
        self.confidence_card.setObjectName("card")
        confidence_layout = QVBoxLayout(self.confidence_card)
        
        # Add header
        confidence_header = QLabel("Confidence Intervals")
        confidence_header.setFont(QFont("Segoe UI", 12, QFont.Bold))
        confidence_header.setStyleSheet("color: white;")
        confidence_layout.addWidget(confidence_header)
        
        # Add description
        confidence_desc = QLabel("Calculate and visualize confidence intervals for predictions")
        confidence_desc.setWordWrap(True)
        confidence_desc.setStyleSheet("color: rgba(255, 255, 255, 0.7);")
        confidence_layout.addWidget(confidence_desc)
        
        # Add combo box for confidence interval type
        confidence_combo_label = QLabel("Select Type:")
        confidence_combo_label.setStyleSheet("color: white; margin-top: 10px;")
        confidence_layout.addWidget(confidence_combo_label)
        
        self.comboPie_Confidence = QComboBox()
        self.comboPie_Confidence.addItem("Exam Score vs Study Hours")
        self.comboPie_Confidence.addItem("Attendance vs Sleep Hours")
        self.comboPie_Confidence.setMinimumHeight(38)
        confidence_layout.addWidget(self.comboPie_Confidence)
        
        # Connect combo box change event to update the hours label
        self.comboPie_Confidence.currentIndexChanged.connect(self.update_confidence_hours_label)
        
        # Add a spin box for input hours
        self.hours_layout = QHBoxLayout()
        self.hours_label = QLabel("Study Hours:")
        self.hours_label.setStyleSheet("color: white; margin-top: 10px;")
        
        self.spinBox_km = QDoubleSpinBox()  # This is the missing spinbox
        self.spinBox_km.setRange(0, 10)  # Set range for hours
        self.spinBox_km.setValue(4)  # Set default value
        self.spinBox_km.setSingleStep(0.5)  # Set step size
        self.spinBox_km.setDecimals(1)  # Show one decimal place
        self.spinBox_km.setMinimumHeight(38)
        
        self.hours_layout.addWidget(self.hours_label)
        self.hours_layout.addWidget(self.spinBox_km)
        confidence_layout.addLayout(self.hours_layout)
        
        # Add search button
        self.SearchPie_Confidence = QPushButton("Run Analysis")
        self.SearchPie_Confidence.setMinimumHeight(38)
        confidence_layout.addWidget(self.SearchPie_Confidence)
        
        # Add this card to the sidebar
        self.cards_layout.addWidget(self.confidence_card)
        
        # Add shadow effect to the card
        confidence_shadow = QGraphicsDropShadowEffect()
        confidence_shadow.setBlurRadius(15)
        confidence_shadow.setColor(QColor(0, 0, 0, 80))
        confidence_shadow.setOffset(0, 5)
        self.confidence_card.setGraphicsEffect(confidence_shadow)
    
    def update_confidence_hours_label(self, index):
        # Update the hours label based on the selected interval type
        if index == 0:  # Exam Score vs Study Hours
            self.hours_label.setText("Study Hours:")
        else:  # Attendance vs Sleep Hours
            self.hours_label.setText("Sleep Hours:")
    
    def create_data_table_card(self):
        # Create a card for data table view
        self.data_table_card = QFrame()
        self.data_table_card.setObjectName("card")
        data_table_layout = QVBoxLayout(self.data_table_card)
        
        # Add header
        data_table_header = QLabel("Data Table View")
        data_table_header.setFont(QFont("Segoe UI", 12, QFont.Bold))
        data_table_header.setStyleSheet("color: white;")
        data_table_layout.addWidget(data_table_header)
        
        # Add description
        data_table_desc = QLabel("Display and explore the dataset in a tabular format")
        data_table_desc.setWordWrap(True)
        data_table_desc.setStyleSheet("color: rgba(255, 255, 255, 0.7);")
        data_table_layout.addWidget(data_table_desc)
        
        # Add filter options
        filter_label = QLabel("Filter by:")
        filter_label.setStyleSheet("color: white; margin-top: 10px;")
        data_table_layout.addWidget(filter_label)
        
        self.data_table_filter_combo = QComboBox()
        self.data_table_filter_combo.addItem("All Data")
        self.data_table_filter_combo.addItem("High Performers (Exam > 80)")
        self.data_table_filter_combo.addItem("Study Hours > 4")
        self.data_table_filter_combo.addItem("Sleep Hours > 7")
        self.data_table_filter_combo.setMinimumHeight(38)
        data_table_layout.addWidget(self.data_table_filter_combo)
        
        # Add search button
        self.data_table_button = QPushButton("Show Data Table")
        self.data_table_button.setMinimumHeight(38)
        data_table_layout.addWidget(self.data_table_button)
        
        # Add this card to the sidebar
        self.cards_layout.addWidget(self.data_table_card)
        
        # Add shadow effect to the card
        data_table_shadow = QGraphicsDropShadowEffect()
        data_table_shadow.setBlurRadius(15)
        data_table_shadow.setColor(QColor(0, 0, 0, 80))
        data_table_shadow.setOffset(0, 5)
        self.data_table_card.setGraphicsEffect(data_table_shadow)
    
    def create_footer(self):
        footer_frame = QFrame()
        footer_frame.setObjectName("footer")
        footer_layout = QHBoxLayout(footer_frame)
        footer_layout.setContentsMargins(15, 10, 15, 10)
        
        # Contributors
        contributors = QLabel("Project by: MUHAMMAD RAMEEZ (23F-0636) | MALIK KAMRAN ALI (23F-0674)")
        contributors.setStyleSheet("font-size: 10px; color: rgba(255, 255, 255, 0.7);")
        footer_layout.addWidget(contributors)
        
        # Create social links layout
        social_layout = QHBoxLayout()
        social_layout.setSpacing(10)
        
        # GitHub button with link
        github_button = QPushButton("GitHub")
        github_button.setFixedSize(80, 25)
        github_button.setStyleSheet("""
            QPushButton {
                background-color: #333;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #444;
            }
        """)
        github_button.clicked.connect(lambda: self.open_url("https://github.com/rameez2005"))
        social_layout.addWidget(github_button)
        
        # LinkedIn button with link
        linkedin_button = QPushButton("LinkedIn")
        linkedin_button.setFixedSize(80, 25)
        linkedin_button.setStyleSheet("""
            QPushButton {
                background-color: #0077B5;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #0069A8;
            }
        """)
        linkedin_button.clicked.connect(lambda: self.open_url("https://www.linkedin.com/in/rameez2005/"))
        social_layout.addWidget(linkedin_button)
        
        # Add social layout to footer
        footer_layout.addLayout(social_layout)
        
        # Apply shadow effect to footer
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 60))
        shadow.setOffset(0, 3)
        footer_frame.setGraphicsEffect(shadow)
        
        self.main_layout.addWidget(footer_frame)
    
    def show_loading(self):
        self.loading_overlay.show()
        self.loading_dots = 0
        self.loading_timer.start(500)  # Update every 500ms
    
    def hide_loading(self):
        self.loading_timer.stop()
        self.loading_overlay.hide()
    
    def update_loading_animation(self):
        self.loading_dots = (self.loading_dots + 1) % 4
        dots = "." * self.loading_dots
        self.loading_label.setText(f"Processing{dots}")
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Update loading overlay size when window is resized
        self.loading_overlay.setGeometry(0, 0, self.width(), self.height())
        self.loading_label.setGeometry(0, 0, self.width(), self.height())
    
    def open_url(self, url):
        """Open a URL in the default web browser"""
        import webbrowser
        webbrowser.open(url) 