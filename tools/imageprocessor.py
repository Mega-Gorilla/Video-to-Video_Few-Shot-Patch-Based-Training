import sys
import os
from PIL import Image
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                            QSpinBox, QMessageBox, QComboBox, QDoubleSpinBox,
                            QStackedWidget)
from PyQt6.QtCore import Qt

class ImageProcessor:
    def __init__(self):
        self.dataset_dir = ""
        self.output_root_dir = ""
        
    def process_images(self, size_mode, size_value):
        """画像の処理を実行する"""
        if not all([self.dataset_dir, self.output_root_dir]):
            raise ValueError("ディレクトリが設定されていません")

        input_dir = os.path.join(self.dataset_dir, 'input')
        dataset_output_dir = os.path.join(self.dataset_dir, 'output')
        
        new_input_dir = os.path.join(self.output_root_dir, 'input')
        new_output_dir = os.path.join(self.output_root_dir, 'output')
        new_mask_dir = os.path.join(self.output_root_dir, 'mask')
        
        os.makedirs(new_input_dir, exist_ok=True)
        os.makedirs(new_output_dir, exist_ok=True)
        os.makedirs(new_mask_dir, exist_ok=True)

        png_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.png')]
        
        for png_file in png_files:
            # PNG画像を読み込む
            png_path = os.path.join(input_dir, png_file)
            png_img = Image.open(png_path)
            
            # サイズモードに応じてリサイズパラメータを計算
            if size_mode == "width":
                target_width = size_value
                target_height = int(target_width * png_img.height / png_img.width)
            else:  # scale
                target_width = int(png_img.width * size_value)
                target_height = int(png_img.height * size_value)
                
            # PNG画像をリサイズして保存
            png_resized = png_img.resize((target_width, target_height), Image.Resampling.LANCZOS)
            png_output_path = os.path.join(new_input_dir, png_file)
            png_resized.save(png_output_path, 'PNG')
            
            # アルファチャンネルからマスクを作成
            if png_img.mode == 'RGBA':
                alpha = png_img.split()[3]
                mask_resized = alpha.resize((target_width, target_height), Image.Resampling.LANCZOS)
                
                mask_filename = os.path.splitext(png_file)[0] + '.jpg'
                mask_path = os.path.join(new_mask_dir, mask_filename)
                mask_resized.convert('RGB').save(mask_path, 'JPEG', quality=95)
            
            # 対応する出力画像を処理
            jpg_filename = os.path.splitext(png_file)[0] + '.png'
            jpg_path = os.path.join(dataset_output_dir, jpg_filename)
            
            if os.path.exists(jpg_path):
                jpg_img = Image.open(jpg_path)
                jpg_resized = jpg_img.resize((target_width, target_height), Image.Resampling.LANCZOS)
                
                output_filename = os.path.splitext(png_file)[0] + '.jpg'
                output_path = os.path.join(new_output_dir, output_filename)
                jpg_resized.save(output_path, 'JPEG', quality=95)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.processor = ImageProcessor()
        self.init_ui()
        
    def init_ui(self):
        """UIの初期化"""
        self.setWindowTitle('画像処理ツール')
        self.setMinimumWidth(600)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # データセットディレクトリ選択
        dataset_layout = QHBoxLayout()
        self.dataset_label = QLabel('データセットフォルダー: 未選択')
        dataset_btn = QPushButton('選択')
        dataset_btn.clicked.connect(lambda: self.select_directory('dataset'))
        dataset_layout.addWidget(self.dataset_label)
        dataset_layout.addWidget(dataset_btn)
        layout.addLayout(dataset_layout)
        
        # 出力ルートディレクトリ選択
        output_layout = QHBoxLayout()
        self.output_label = QLabel('出力先フォルダー: 未選択')
        output_btn = QPushButton('選択')
        output_btn.clicked.connect(lambda: self.select_directory('output'))
        output_layout.addWidget(self.output_label)
        output_layout.addWidget(output_btn)
        layout.addLayout(output_layout)
        
        # サイズ設定モード選択
        size_mode_layout = QHBoxLayout()
        size_mode_layout.addWidget(QLabel('サイズ指定方式:'))
        self.size_mode_combo = QComboBox()
        self.size_mode_combo.addItems(['幅指定', '倍率指定'])
        self.size_mode_combo.currentIndexChanged.connect(self.on_size_mode_changed)
        size_mode_layout.addWidget(self.size_mode_combo)
        layout.addLayout(size_mode_layout)
        
        # サイズ入力スタックウィジェット
        self.size_stack = QStackedWidget()
        
        # 幅指定用ウィジェット
        width_widget = QWidget()
        width_layout = QHBoxLayout(width_widget)
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 10000)
        self.width_spin.setValue(1024)
        width_layout.addWidget(QLabel('幅:'))
        width_layout.addWidget(self.width_spin)
        width_layout.addWidget(QLabel('px'))
        width_layout.addStretch()
        
        # 倍率指定用ウィジェット
        scale_widget = QWidget()
        scale_layout = QHBoxLayout(scale_widget)
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0.1, 10.0)
        self.scale_spin.setValue(1.0)
        self.scale_spin.setSingleStep(0.1)
        scale_layout.addWidget(QLabel('倍率:'))
        scale_layout.addWidget(self.scale_spin)
        scale_layout.addWidget(QLabel('倍'))
        scale_layout.addStretch()
        
        self.size_stack.addWidget(width_widget)
        self.size_stack.addWidget(scale_widget)
        layout.addWidget(self.size_stack)
        
        # 実行ボタン
        process_btn = QPushButton('処理実行')
        process_btn.clicked.connect(self.process_images)
        layout.addWidget(process_btn)
        
    def on_size_mode_changed(self, index):
        """サイズ指定モードが変更されたときの処理"""
        self.size_stack.setCurrentIndex(index)
        
    def select_directory(self, dir_type):
        """ディレクトリ選択ダイアログを表示"""
        directory = QFileDialog.getExistingDirectory(self, 'フォルダーを選択')
        if directory:
            if dir_type == 'dataset':
                self.processor.dataset_dir = directory
                self.dataset_label.setText(f'データセットフォルダー: {directory}')
            elif dir_type == 'output':
                self.processor.output_root_dir = directory
                self.output_label.setText(f'出力先フォルダー: {directory}')
    
    def process_images(self):
        """画像処理を実行"""
        try:
            input_dir = os.path.join(self.processor.dataset_dir, 'input')
            output_dir = os.path.join(self.processor.dataset_dir, 'output')
            
            if not os.path.exists(input_dir) or not os.path.exists(output_dir):
                raise ValueError("データセットフォルダー内にinputとoutputフォルダーが必要です")
            
            # サイズモードと値を取得
            size_mode = "width" if self.size_mode_combo.currentIndex() == 0 else "scale"
            size_value = self.width_spin.value() if size_mode == "width" else self.scale_spin.value()
                
            self.processor.process_images(size_mode, size_value)
            QMessageBox.information(self, '完了', '画像処理が完了しました。')
        except Exception as e:
            QMessageBox.critical(self, 'エラー', f'処理中にエラーが発生しました: {str(e)}')

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()