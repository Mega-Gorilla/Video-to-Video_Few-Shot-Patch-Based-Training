import sys
import os
from PIL import Image
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                            QSpinBox, QMessageBox, QComboBox, QDoubleSpinBox,
                            QStackedWidget, QColorDialog, QProgressBar)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

class ImageProcessorThread(QThread):
    """画像処理を行うスレッド"""
    progress_updated = pyqtSignal(int)  # 進捗更新シグナル
    finished = pyqtSignal()  # 完了シグナル
    error = pyqtSignal(str)  # エラーシグナル
    
    def __init__(self, dataset_dir, output_root_dir, size_mode, size_value, bg_color):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.output_root_dir = output_root_dir
        self.size_mode = size_mode
        self.size_value = size_value
        self.bg_color = bg_color
        
    def process_single_image(self, args):
        """1つの画像を処理"""
        try:
            png_file, input_dir, dataset_output_dir, new_input_dir, new_output_dir, new_mask_dir = args
            
            # PNG画像を読み込む
            png_path = os.path.join(input_dir, png_file)
            png_img = Image.open(png_path)
            
            # サイズモードに応じてリサイズパラメータを計算
            if self.size_mode == "width":
                target_width = self.size_value
                target_height = int(target_width * png_img.height / png_img.width)
            else:  # scale
                target_width = int(png_img.width * self.size_value)
                target_height = int(png_img.height * self.size_value)

            # アルファチャンネルの処理
            if png_img.mode == 'RGBA':
                # マスク画像の作成
                alpha = png_img.split()[3]
                mask_resized = alpha.resize((target_width, target_height), Image.Resampling.LANCZOS)
                mask_filename = os.path.splitext(png_file)[0] + '.jpg'
                mask_path = os.path.join(new_mask_dir, mask_filename)
                mask_resized.convert('RGB').save(mask_path, 'JPEG', quality=95)
                
                # 背景色の設定
                bg_color_rgb = (self.bg_color.red(), self.bg_color.green(), self.bg_color.blue())
                bg = Image.new('RGB', png_img.size, bg_color_rgb)
                bg.paste(png_img, mask=png_img.split()[3])
                png_img = bg
            
            # リサイズして保存
            png_resized = png_img.resize((target_width, target_height), Image.Resampling.LANCZOS)
            png_output_path = os.path.join(new_input_dir, os.path.splitext(png_file)[0] + '.jpg')
            png_resized.convert('RGB').save(png_output_path, 'JPEG', quality=95)
            
            # 対応する出力画像を処理（存在する場合のみ）
            jpg_filename = os.path.splitext(png_file)[0] + '.png'
            jpg_path = os.path.join(dataset_output_dir, jpg_filename)
            
            if os.path.exists(jpg_path):
                jpg_img = Image.open(jpg_path)
                jpg_resized = jpg_img.resize((target_width, target_height), Image.Resampling.LANCZOS)
                
                output_filename = os.path.splitext(png_file)[0] + '.jpg'
                output_path = os.path.join(new_output_dir, output_filename)
                jpg_resized.save(output_path, 'JPEG', quality=95)
                
            return True
        except Exception as e:
            print(f"Error processing {png_file}: {str(e)}")
            return False

    def run(self):
        """スレッドのメイン処理"""
        try:
            input_dir = os.path.join(self.dataset_dir, 'input')
            dataset_output_dir = os.path.join(self.dataset_dir, 'output')
            
            new_input_dir = os.path.join(self.output_root_dir, 'input')
            new_output_dir = os.path.join(self.output_root_dir, 'output')
            new_mask_dir = os.path.join(self.output_root_dir, 'mask')
            
            os.makedirs(new_input_dir, exist_ok=True)
            os.makedirs(new_output_dir, exist_ok=True)
            os.makedirs(new_mask_dir, exist_ok=True)

            png_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.png')]
            total_files = len(png_files)
            
            # 並列処理用の引数リストを作成
            args_list = [(png_file, input_dir, dataset_output_dir, new_input_dir, new_output_dir, new_mask_dir)
                        for png_file in png_files]
            
            processed_count = 0
            # ThreadPoolExecutorを使用して並列処理
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(self.process_single_image, args) 
                          for args in args_list]
                
                for future in concurrent.futures.as_completed(futures):
                    processed_count += 1
                    progress = int((processed_count / total_files) * 100)
                    self.progress_updated.emit(progress)
            
            self.finished.emit()
            
        except Exception as e:
            self.error.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.bg_color = QColor(255, 255, 255)
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

        # 背景色選択
        bg_color_layout = QHBoxLayout()
        bg_color_layout.addWidget(QLabel('背景色:'))
        self.color_btn = QPushButton()
        self.color_btn.setFixedSize(50, 25)
        self.update_color_button()
        self.color_btn.clicked.connect(self.select_color)
        bg_color_layout.addWidget(self.color_btn)
        bg_color_layout.addStretch()
        layout.addLayout(bg_color_layout)
        
        # プログレスバー
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # 実行ボタン
        self.process_btn = QPushButton('処理実行')
        self.process_btn.clicked.connect(self.process_images)
        layout.addWidget(self.process_btn)
        
    def update_color_button(self):
        """カラーボタンの背景色を更新"""
        self.color_btn.setStyleSheet(
            f'background-color: rgb({self.bg_color.red()}, {self.bg_color.green()}, {self.bg_color.blue()}); '
            'border: 1px solid black;'
        )
        
    def select_color(self):
        """カラーパレットを表示して背景色を選択"""
        color = QColorDialog.getColor(self.bg_color, self)
        if color.isValid():
            self.bg_color = color
            self.update_color_button()
        
    def on_size_mode_changed(self, index):
        """サイズ指定モードが変更されたときの処理"""
        self.size_stack.setCurrentIndex(index)
        
    def select_directory(self, dir_type):
        """ディレクトリ選択ダイアログを表示"""
        directory = QFileDialog.getExistingDirectory(self, 'フォルダーを選択')
        if directory:
            if dir_type == 'dataset':
                self.dataset_dir = directory
                self.dataset_label.setText(f'データセットフォルダー: {directory}')
            elif dir_type == 'output':
                self.output_dir = directory
                self.output_label.setText(f'出力先フォルダー: {directory}')
    
    def process_images(self):
        """画像処理を実行"""
        try:
            # 入力チェック
            if not hasattr(self, 'dataset_dir') or not hasattr(self, 'output_dir'):
                raise ValueError("データセットフォルダーと出力先フォルダーを選択してください")
                
            input_dir = os.path.join(self.dataset_dir, 'input')
            if not os.path.exists(input_dir):
                raise ValueError("データセットフォルダー内にinputフォルダーが必要です")
            
            # サイズモードと値を取得
            size_mode = "width" if self.size_mode_combo.currentIndex() == 0 else "scale"
            size_value = self.width_spin.value() if size_mode == "width" else self.scale_spin.value()
            
            # UIの更新
            self.process_btn.setEnabled(False)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            # 処理スレッドの作成と開始
            self.processor_thread = ImageProcessorThread(
                self.dataset_dir,
                self.output_dir,
                size_mode,
                size_value,
                self.bg_color
            )
            
            # シグナルの接続
            self.processor_thread.progress_updated.connect(self.update_progress)
            self.processor_thread.finished.connect(self.process_completed)
            self.processor_thread.error.connect(self.process_error)
            
            # スレッド開始
            self.processor_thread.start()
            
        except Exception as e:
            QMessageBox.critical(self, 'エラー', f'処理の開始に失敗しました: {str(e)}')
            self.process_btn.setEnabled(True)
            self.progress_bar.setVisible(False)
            
    def update_progress(self, value):
        """進捗バーの更新"""
        self.progress_bar.setValue(value)
        
    def process_completed(self):
        """処理完了時の処理"""
        self.process_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.information(self, '完了', '画像処理が完了しました。')
        
    def process_error(self, error_message):
        """エラー発生時の処理"""
        self.process_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, 'エラー', f'処理中にエラーが発生しました: {error_message}')

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()