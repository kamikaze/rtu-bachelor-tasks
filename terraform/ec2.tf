# EC2 instances

resource "aws_spot_instance_request" "k3s_master_spot_ec2_instance" {
  count = var.ec2_enabled ? 1 : 0
  depends_on = [
    aws_subnet.rtu_bachelor_subnet_eu_north_1a,
    aws_security_group.rtu_bachelor_ssh_sg
  ]
  ami                            = "ami-0c76bf0e69c8a6228"
  spot_price                     = "0.01"
  spot_type                      = "persistent"
  instance_type                  = "t4g.small"
  instance_interruption_behavior = "stop"
  wait_for_fulfillment           = true
  key_name                       = "cl-dev-keypair"
  subnet_id                      = aws_subnet.rtu_bachelor_subnet_eu_north_1a.id
  vpc_security_group_ids         = [aws_security_group.rtu_bachelor_ssh_sg.id]
  associate_public_ip_address    = true

  tags = {
    Arch         = "arm64"
    HasGPU       = "no"
    Name         = "k3s-master1",
    Organization = "RTU"
    SpotPrice    = "yes"
  }
}

#resource "aws_spot_instance_request" "rtu_cpu_spot_ec2_instance" {
#  count = var.ec2-enabled ? 1 : 0
#  depends_on = [
#    aws_subnet.rtu_bachelor_subnet_eu_north_1a,
#    aws_security_group.rtu_bachelor_ssh_sg,
#    aws_spot_instance_request.k3s_master_spot_ec2_instance
#  ]
#  ami                            = "ami-0c76bf0e69c8a6228"
#  spot_price                     = "1.00"
#  spot_type                      = "persistent"
#  instance_type                  = "c6g.16xlarge"
#  instance_interruption_behavior = "stop"
#  wait_for_fulfillment           = true
#  key_name                       = "cl-dev-keypair"
#  subnet_id                      = aws_subnet.rtu_bachelor_subnet_eu_north_1a.id
#  vpc_security_group_ids         = [aws_security_group.rtu_bachelor_ssh_sg.id]
#  associate_public_ip_address    = true
#
#  tags = {
#    Arch         = "arm64"
#    HasGPU       = "no"
#    Name         = "rtu-cpu-monster1",
#    Organization = "RTU"
#    SpotPrice    = "yes"
#  }
#}

#resource "aws_spot_instance_request" "rtu_gpu_spot_ec2_instance" {
#  count = var.ec2-enabled ? 1 : 0
#  depends_on = [
#    aws_subnet.rtu_bachelor_subnet_eu_north_1a,
#    aws_security_group.rtu_bachelor_ssh_sg,
#    aws_spot_instance_request.k3s_master_spot_ec2_instance
#  ]
#  ami                            = "ami-0bb935e4614c12d86"
#  spot_price                     = "1.75"
#  spot_type                      = "persistent"
#  instance_type                  = "g5.16xlarge"
#  instance_interruption_behavior = "stop"
#  wait_for_fulfillment           = true
#  key_name                       = "cl-dev-keypair"
#  subnet_id                      = aws_subnet.rtu_bachelor_subnet_eu_north_1a.id
#  vpc_security_group_ids         = [aws_security_group.rtu_bachelor_ssh_sg.id]
#  associate_public_ip_address    = true
#
#  tags = {
#    Arch         = "amd64"
#    HasGPU       = "yes"
#    Name         = "rtu-gpu-monster1",
#    Organization = "RTU"
#    SpotPrice    = "yes"
#  }
#}
