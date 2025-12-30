import os
import shutil
import re
from django.core.management.base import BaseCommand
from django.conf import settings
from cvat.apps.engine.models import Job, FrameLogitMeta

class Command(BaseCommand):
    help = 'Import logit maps từ folder máy tính vào CVAT Backend'

    def add_arguments(self, parser):
        parser.add_argument('job_id', type=int, help='ID của Job muốn gán logit')
        parser.add_argument('source_dir', type=str, help='Đường dẫn folder chứa ảnh slice_xxxx.png')

    def handle(self, *args, **options):
        job_id = options['job_id']
        source_dir = options['source_dir']

        # 1. Kiểm tra Job có tồn tại không
        try:
            job = Job.objects.get(pk=job_id)
            self.stdout.write(f"Found Job ID: {job_id}")
        except Job.DoesNotExist:
            self.stdout.write(self.style.ERROR(f'Job {job_id} không tồn tại!'))
            return

        # 2. Tạo đường dẫn đích (Nơi lưu trữ chuẩn của CVAT)
        # File sẽ được lưu tại: cvat_data/logits/jobs/<job_id>/
        relative_dest_dir = os.path.join('logits', 'jobs', str(job_id))
        abs_dest_dir = os.path.join(settings.DATA_ROOT, relative_dest_dir)

        if not os.path.exists(abs_dest_dir):
            os.makedirs(abs_dest_dir)
            self.stdout.write(f"Đã tạo thư mục: {abs_dest_dir}")

        # 3. Quét file và xử lý
        files = sorted([f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        count = 0
        frame_mapping = {}  # Để debug
        for filename in files:
            # Parse số frame từ tên file (Ví dụ: slice_0001.png -> frame 0)
            # Logic: Tìm nhóm số cuối cùng trong tên file
            match = re.search(r'(\d+)', filename)
            if not match:
                self.stdout.write(self.style.WARNING(f"Bỏ qua {filename}: Không tìm thấy số frame"))
                continue

            # Giả sử file bắt đầu từ 1 (slice_0001) tương ứng frame 0 của CVAT
            file_num = int(match.group(1))
            frame_number = file_num - 1

            if frame_number < 0:
                self.stdout.write(self.style.WARNING(f"Bỏ qua {filename}: Frame number < 0 ({frame_number})"))
                continue

            # A. COPY FILE VÀO DATA_ROOT
            src_path = os.path.join(source_dir, filename)
            dest_path = os.path.join(abs_dest_dir, filename)
            try:
            shutil.copy2(src_path, dest_path)
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Lỗi copy file {filename}: {e}"))
                continue

            # B. TẠO BẢN GHI DATABASE (FrameLogitMeta)
            # Đường dẫn lưu trong DB là đường dẫn tương đối
            relative_path_file = os.path.join(relative_dest_dir, filename)
            mime = 'image/png' if filename.endswith('.png') else 'image/jpeg'

            try:
            obj, created = FrameLogitMeta.objects.update_or_create(
                job=job,
                frame=frame_number,
                defaults={
                    'relative_path': relative_path_file,
                    'mime_type': mime
                }
            )
            status = "Mới" if created else "Update"
                frame_mapping[frame_number] = filename
                # Log đặc biệt cho frame 0
                if frame_number == 0:
                    self.stdout.write(self.style.SUCCESS(f"✅ Frame 0: {status} ({filename}) -> {relative_path_file}"))
            count += 1
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Lỗi tạo bản ghi DB cho {filename} (frame {frame_number}): {e}"))
                continue

        self.stdout.write(self.style.SUCCESS(f'XONG! Đã nạp {count} logit maps vào Job {job_id}.'))
        if frame_mapping:
            frames_list = sorted(frame_mapping.keys())
            self.stdout.write(f'   Frames imported: {frames_list[0]} to {frames_list[-1]} (total: {len(frames_list)})')
            if 0 in frame_mapping:
                self.stdout.write(self.style.SUCCESS(f'   ✅ Frame 0: {frame_mapping[0]}'))
            else:
                self.stdout.write(self.style.WARNING(f'   ⚠️  Frame 0 KHÔNG có trong danh sách!'))