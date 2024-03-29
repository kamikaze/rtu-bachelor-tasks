# Security groups

resource "aws_security_group" "ssh_sg" {
  name        = "${var.project}-ssh-sg"
  description = "SSH security group"
  vpc_id      = aws_vpc.main_vpc.id

  tags = {
    Name         = "${var.project}-ssh-sg",
    Organization = var.organization
    Project      = var.project
  }
}

resource "aws_vpc_security_group_egress_rule" "ssh_sg_ipv4_all" {
  security_group_id = aws_security_group.ssh_sg.id
  cidr_ipv4         = "0.0.0.0/0"
  ip_protocol       = "-1"
}

resource "aws_vpc_security_group_egress_rule" "ssh_sg_ipv6_all" {
  security_group_id = aws_security_group.ssh_sg.id
  cidr_ipv6         = "::/0"
  ip_protocol       = "-1"
}

resource "aws_vpc_security_group_ingress_rule" "ssh_sg_ipv4_all" {
  security_group_id = aws_security_group.ssh_sg.id
  cidr_ipv4         = "0.0.0.0/0"
  from_port         = 22
  to_port           = 22
  ip_protocol       = "tcp"
}

resource "aws_vpc_security_group_ingress_rule" "ssh_sg_ipv6_all" {
  security_group_id = aws_security_group.ssh_sg.id
  cidr_ipv6         = "::/0"
  from_port         = 22
  to_port           = 22
  ip_protocol       = "tcp"
}

resource "aws_security_group" "postgres_sg" {
  name        = "${var.project}-postgres-sg"
  description = "PostgreSQL security group"
  vpc_id      = aws_vpc.main_vpc.id

  tags = {
    Name         = "${var.project}-postgres-sg",
    Organization = var.organization
    Project      = var.project
  }
}

resource "aws_vpc_security_group_egress_rule" "postgres_sg_ipv4_all" {
  security_group_id = aws_security_group.postgres_sg.id
  cidr_ipv4         = "0.0.0.0/0"
  ip_protocol       = "-1"
}

resource "aws_vpc_security_group_egress_rule" "postgres_sg_ipv6_all" {
  security_group_id = aws_security_group.postgres_sg.id
  cidr_ipv6         = "::/0"
  ip_protocol       = "-1"
}

resource "aws_vpc_security_group_ingress_rule" "postgres_sg_ipv4_all" {
  security_group_id = aws_security_group.postgres_sg.id
  cidr_ipv4         = "0.0.0.0/0"
  from_port         = 5432
  to_port           = 5432
  ip_protocol       = "tcp"
}

resource "aws_vpc_security_group_ingress_rule" "postgres_sg_ipv6_all" {
  security_group_id = aws_security_group.postgres_sg.id
  cidr_ipv6         = "::/0"
  from_port         = 5432
  to_port           = 5432
  ip_protocol       = "tcp"
}

resource "aws_security_group" "eks_cluster_sg" {
  count       = var.eks_enabled ? 1 : 0
  name        = "${var.project}-eks-cluster-sg"
  description = "EKS cluster security group"
  vpc_id      = aws_vpc.main_vpc.id

  tags = {
    Name         = "${var.project}-eks-cluster-sg",
    Organization = var.organization
    Project      = var.project
  }
}

resource "aws_vpc_security_group_egress_rule" "eks_cluster_sg_ipv4_all" {
  depends_on = [
    aws_security_group.eks_cluster_sg[0]
  ]
  count             = var.eks_enabled ? 1 : 0
  security_group_id = aws_security_group.eks_cluster_sg[0].id
  cidr_ipv4         = "0.0.0.0/0"
  ip_protocol       = "-1"
}

resource "aws_vpc_security_group_egress_rule" "eks_cluster_sg_ipv6_all" {
  depends_on = [
    aws_security_group.eks_cluster_sg[0]
  ]
  count             = var.eks_enabled ? 1 : 0
  security_group_id = aws_security_group.eks_cluster_sg[0].id
  cidr_ipv6         = "::/0"
  ip_protocol       = "-1"
}

resource "aws_vpc_security_group_ingress_rule" "eks_cluster_sg_ipv4_all" {
  depends_on = [
    aws_security_group.eks_cluster_sg[0]
  ]
  count             = var.eks_enabled ? 1 : 0
  security_group_id = aws_security_group.eks_cluster_sg[0].id
  cidr_ipv4         = "0.0.0.0/0"
  ip_protocol       = "-1"
}

resource "aws_vpc_security_group_ingress_rule" "eks_cluster_sg_ipv6_all" {
  depends_on = [
    aws_security_group.eks_cluster_sg[0]
  ]
  count             = var.eks_enabled ? 1 : 0
  security_group_id = aws_security_group.eks_cluster_sg[0].id
  cidr_ipv6         = "::/0"
  ip_protocol       = "-1"
}

resource "aws_security_group" "eks_node_group_sg" {
  count       = var.eks_enabled ? 1 : 0
  name        = "${var.project}-eks-node-group-sg"
  description = "Security group for EKS Node Group"
  vpc_id      = aws_vpc.main_vpc.id

  tags = {
    Name                                            = "${var.project}-eks-node-group-sg",
    Organization                                    = var.organization
    Project                                         = var.project
    "kubernetes.io/cluster/${var.eks_cluster_name}" = "shared"
  }
}

resource "aws_vpc_security_group_egress_rule" "eks_node_group_sg_ipv4_all" {
  depends_on = [
    aws_security_group.eks_node_group_sg[0]
  ]
  count             = var.eks_enabled ? 1 : 0
  security_group_id = aws_security_group.eks_node_group_sg[0].id
  cidr_ipv4         = "0.0.0.0/0"
  ip_protocol       = "-1"
}

resource "aws_vpc_security_group_egress_rule" "eks_node_group_sg_ipv6_all" {
  depends_on = [
    aws_security_group.eks_node_group_sg[0]
  ]
  count             = var.eks_enabled ? 1 : 0
  security_group_id = aws_security_group.eks_node_group_sg[0].id
  cidr_ipv6         = "::/0"
  ip_protocol       = "-1"
}

resource "aws_vpc_security_group_ingress_rule" "eks_node_group_sg_ipv4_all" {
  depends_on = [
    aws_security_group.eks_node_group_sg[0]
  ]
  count             = var.eks_enabled ? 1 : 0
  security_group_id = aws_security_group.eks_node_group_sg[0].id
  cidr_ipv4         = "0.0.0.0/0"
  ip_protocol       = "-1"
}

resource "aws_vpc_security_group_ingress_rule" "eks_node_group_sg_ipv6_all" {
  depends_on = [
    aws_security_group.eks_node_group_sg[0]
  ]
  count             = var.eks_enabled ? 1 : 0
  security_group_id = aws_security_group.eks_node_group_sg[0].id
  cidr_ipv6         = "::/0"
  ip_protocol       = "-1"
}
