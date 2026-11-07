# TIP swing-up cho con lắc 2 khâu

Task độc lập, tham khảo `triple_tip_swingup/src/cart_triple_pendulum/tasks/tip.py`
và `tasks/triple/tip_swing_up/reward.json`; không thay reward các task cũ.

Góc tuyệt đối: theta = cumsum(q[1:]) - pi; vận tốc tuyệt đối:
omega = cumsum(v[1:]). MJCF dùng q=0 là treo xuống.

R_tip = exp(-0.4 a²) exp(-0.3 |x|)
        product_i [(1+cos(theta_i))/2 * exp(-k_i |omega_i|)]

Chọn k=(0.015, 0.009), lấy hai hệ số đầu của bản 3 khâu; đây là lựa chọn
chuyển thể, chưa được tối ưu riêng cho plant 2 khâu. a là lệnh chuẩn hóa
trước rail guard; x tính bằng mét, omega bằng rad/s.

Reward tổng = R_tip - 0.2 R_tip gate (1-settling) + gate settling centered
              + 3 * precision_center - rail_terminal_cost.
Gate Gaussian dùng hai sai số góc tuyệt đối; settling dùng tốc độ lớn nhất
sau chuẩn hóa của xe và hai khâu tuyệt đối, giống bản triple. Các ngưỡng
capture theo plant-scale của expert swing hiện có. Thêm rail_terminal_cost trọng số -1 theo contract chống kết thúc sớm của project;
chi phí terminal lấy discount từ trainer. Reward không nhân dt.

Giữ reset curriculum của expert_swing_sac (hanging và near-capture), không
cần handoff bank. Mặc định tắt randomization/noise/push cho thí nghiệm nominal;
evaluation vẫn gồm hanging nominal và randomized. SAC reward_scale=1.
Kết quả lưu riêng trong results/real/tip_swing_sac.

Chạy từ thư mục project, bằng môi trường có đủ dependencies:

```bash
python tasks/real/tip_swing_sac/run.py inspect
python tasks/real/tip_swing_sac/run.py train
python tasks/real/tip_swing_sac/run.py evaluate --episodes 32
```

Chỉ số smoke CPU xác nhận đường chạy optimizer/checkpoint/ONNX, không xác
nhận policy đã swing-up hoặc cân bằng thành công. Training dài cần chạy riêng.

Profile SAC local cho run mới: 128 env, 20M transitions, episode 10 s; xem
[EP_TASKS](../../../docs/EP_TASKS.md). Run remote đã khởi chạy giữ nguyên profile cũ.

Yêu cầu mới cho run tiếp theo: giữ mục tiêu đồng thời |x| ≤ 5 mm liên tục 1 s.
Bonus precision_center có gradient dẫn về tâm từ xa và kernel hẹp quanh vùng 5 mm.
Policy cũ chưa được xác nhận theo tiêu chí này; xem docs/EP_TASKS.md.
