# EC2 instances

locals {
  nodes = [1, 2, 3]
}

resource "aws_spot_instance_request" "node" {
  count = var.ec2_enabled ? length(local.nodes) : 0
  depends_on = [
    aws_subnet.public_subnet_a,
    aws_subnet.public_subnet_b,
    aws_subnet.public_subnet_c,
    aws_security_group.ssh_sg
  ]
  ami                            = "ami-04c97e62cb19d53f1"
  spot_price                     = "0.0047"
  spot_type                      = "persistent"
  instance_type                  = "t4g.nano"
  instance_interruption_behavior = "stop"
  wait_for_fulfillment           = true
  key_name                       = var.key_pair_name
  subnet_id                      = aws_subnet.public_subnet_a.id
  vpc_security_group_ids         = [aws_security_group.ssh_sg.id]
  associate_public_ip_address    = true

  tags = {
    Arch         = "arm64"
    HasGPU       = "no"
    Name         = "${var.project}-node-${local.nodes[count.index]}",
    Organization = var.organization
    Project      = var.project
    SpotPrice    = "yes"
  }
}
